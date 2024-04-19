from typing import Optional, Tuple, Literal, Union, List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from trajectory_gpt2 import GPT2Model
from vector_quantize_pytorch import VectorQuantize

from utils import pad,entropy

import einops
from einops.layers.torch import Rearrange
import copy



LRELU_SLOPE = 0.02

'''
This is the skill predictor in LISA
'''
class OptionTransformer(nn.Module):

    """
    This model uses GPT-2 to select options for every horizon-th state
    """

    def __init__(
            self,
            state_dim,
            lang_dim,
            option_dim,
            num_cams,
            hidden_size,
            max_length=None,
            max_ep_len=512,
            **kwargs):
        super().__init__()

        self.option_dim = option_dim
        self.num_cams = num_cams
        self.hidden_size = hidden_size

        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        self.state_dim = state_dim
        self.max_length = max_length
        self.output_attentions = kwargs["output_attentions"]

        #need update
        # if isinstance(state_dim, tuple):
        #     if state_dim[-1] == 3:
        #         # RLBench
        #         self.embed_state = Encoder(hidden_size=hidden_size, ch=3)
        #     else:
        #         # LORL Franka
        #         self.embed_state = Encoder(hidden_size=hidden_size, ch=12)

        # else:
        #     self.embed_state = nn.Linear(self.state_dim, hidden_size)

        self.embed_lang = nn.Linear(lang_dim, hidden_size)

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.cam_embedding = nn.Embedding(self.num_cams, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)
        self.predict_options = torch.nn.Linear(hidden_size, self.option_dim)

    def forward(self, word_embeddings, state_embeddings, timesteps, attention_mask, **kwargs):
        # batch_size, seq_length,num_cam,H,W,C = states.shape
        batch_size, seq_length,num_cam,_ = state_embeddings.shape

        if attention_mask is None:
            raise ValueError('Should not have attention_mask NONE')
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length))

        # state_embeddings = self.embed_state(states)


        lang_embeddings = self.embed_lang(word_embeddings)
        time_embeddings = self.embed_timestep(timesteps)
        cam_embeddings = self.cam_embedding(torch.arange(self.num_cams).to(state_embeddings.device))

        # time embeddings and camera embeddings are treated similar to positional embeddings
        # (batch_size, seq_length,num_cam, hidden)
        state_embeddings = state_embeddings + einops.rearrange(time_embeddings, 'b t c -> b t 1 c') + \
          einops.rearrange(cam_embeddings, 'n c -> 1 1 n c')  
        # lang_embeddings = einops.rearrange(lang_embeddings, 'b l c -> b l 1 c').repeat([1,1,num_cam,1])
        lang_embeddings = einops.rearrange(lang_embeddings, 'b c -> b 1 1 c').repeat([1,1,num_cam,1])
        num_tokens = lang_embeddings.shape[1]

        # batch_size, num_tokens+seq_length,num_cam, hidden
        lang_and_inputs = torch.cat([lang_embeddings, state_embeddings], dim=1) 

        # batch_size, (num_tokens+seq_length)*num_cam, hidden
        lang_and_inputs = einops.rearrange(lang_and_inputs, 'b t n c -> b (t n) c')
        
        # LAYERNORM AFTER LANGUAGE,  batch_size, (num_tokens+seq_length)*num_cam, hidden
        lang_and_inputs = self.embed_ln(lang_and_inputs)

        lang_attn_mask = torch.cat([torch.ones((batch_size, num_tokens), device=state_embeddings.device), attention_mask],dim=1).repeat([1,num_cam])

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=lang_and_inputs,
            attention_mask=lang_attn_mask,
        )

        x = transformer_outputs['last_hidden_state']

        # (batch_size, seq_length,num_cam, hidden)
        x = torch.reshape(x,[batch_size,num_tokens+seq_length,num_cam,self.hidden_size])
        lang_out = x[:, :num_tokens, :,:].reshape(batch_size, num_tokens*num_cam, self.hidden_size)
        traj_out = x[:, num_tokens:, :,:].reshape(batch_size, seq_length*num_cam, self.hidden_size)

        # get predictions
        # predict option logits given state
        option_preds = self.predict_options(traj_out)
        option_preds = einops.rearrange(option_preds,'b (t n) c -> b t n c',t=seq_length,n=num_cam)

        if self.output_attentions:
            attentions = transformer_outputs[-1]
            return option_preds, attentions

        return option_preds, None



class OptionSelector(nn.Module):

    """
    This model takes in the language embedding and the state to output a z from a categorical distribution
    Use the VQ trick to pick an option z
    """

    def __init__(
            self, state_dim, num_options, option_dim,num_cams, lang_dim, num_hidden=None, hidden_size=128,
            option_transformer=None, codebook_dim=16, use_vq=True, kmeans_init=False,
            commitment_weight=0.25, **kwargs):

        # option_dim and codebook_dim are different because of the way the VQ package is designed
        # if they are different, then there is a projection operation that happens inside the VQ layer

        super().__init__()

        if num_hidden is not None:
            assert num_hidden >= 2, "At least two hidden layers!"

        self.state_dim = state_dim
        self.option_dim = option_dim
        self.use_vq = use_vq
        self.num_options = num_options
        
        self.hidden_size = hidden_size


        option_transformer_args = {'state_dim': state_dim,
                                    'lang_dim': lang_dim,
                                    'option_dim': option_dim,
                                    'num_cams': num_cams,
                                    'hidden_size': hidden_size,
                                    'max_length': option_transformer.max_length,
                                    'max_ep_len': option_transformer.max_ep_len,
                                    'n_layer': option_transformer.n_layer,
                                    'n_head': option_transformer.n_head,
                                    'n_inner': 4*hidden_size,
                                    'activation_function': option_transformer.activation_function,
                                    'n_positions': option_transformer.n_positions,
                                    'resid_pdrop': option_transformer.dropout,
                                    'attn_pdrop': option_transformer.dropout,
                                    'output_attentions': True  # option_transformer.output_attention,
                                    }
        self.option_dt = OptionTransformer(**option_transformer_args)

        self.Z = VectorQuantize(
            dim=option_dim,
            codebook_dim=codebook_dim,       # codebook vector size
            codebook_size=num_options,     # codebook size
            decay=0.99,             # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight=commitment_weight,   # the weight on the commitment loss
            kmeans_init=kmeans_init,   # use kmeans init
            cpc=False,
            # threshold_ema_dead_code=2,  # should actively replace any codes that have an exponential moving average cluster size less than 2
            use_cosine_sim=False   # l2 normalize the codes
        )

        self.fuse_option = nn.Sequential(
          *dense_layer(hidden_size*num_cams,hidden_size),
          *dense_layer(hidden_size,hidden_size, apply_activation=False)
        )

    def penalize_similarity(self,extr_opt, keyframe_ops, margin=1.0):
        """
        Penalize similarity using a margin-based loss.
        
        Args:
        - extracted_opt (torch.Tensor): Tensor representing extracted options.
        - keyframe_options (torch.Tensor): Tensor representing keyframe options.
        - margin (float): The margin threshold to enforce between extracted options and keyframe options.
        
        Returns:
        - loss (torch.Tensor): The computed margin-based loss.
        """
        # Ensure extracted_opt is broadcasted/repeated to match keyframe_options shape
        ba, keyframe_len, _ = keyframe_ops.shape
        expanded_opt = einops.repeat(extr_opt, 'c -> ba l c', ba=ba, l=keyframe_len)
        
        # Compute the pairwise Euclidean distance between expanded_opt and keyframe_options
        distance = F.pairwise_distance(expanded_opt, keyframe_ops, keepdim=True)
        
        # Compute margin-based loss
        # This loss is positive when distance < margin, encouraging the model to increase the distance
        loss = F.relu(margin - distance).mean()
        
        return loss


    def forward(self, word_embeddings, state_embeddings, timesteps=None, attention_mask=None, **kwargs):
        #output:    option shape B, T, hidden
        #           index shape B, T
        commitment_loss_weight = 1
        option_preds = self.option_dt(word_embeddings, state_embeddings, timesteps, attention_mask)[0]
        # need update, intergrate keyframe

        # option_preds = option_preds[:, :, :]
        option_preds = einops.rearrange(option_preds,'b t n c ->b t (n c)')
        option_preds = self.fuse_option(option_preds)
        options, indices, commitment_loss = self.Z(option_preds)
        ori_opt = options.clone().detach()

        for ba in range(options.shape[0]):
          _,index = torch.mode(options[ba, :5, :],0)
          extracted_index = torch.argmax(torch.bincount(index))
          extracted_option = options[ba,extracted_index,:]
          # extracted_option = options[ba,0,:]
          options[ba,1:5,:] = extracted_option.clone().detach()
          options[ba,0,:] = extracted_option.clone()
        penal_sim = self.penalize_similarity(extr_opt=extracted_option, keyframe_ops=options[:,5:,:])
        
        commitment_loss = commitment_loss + penal_sim
        # commitment_loss = commitment_loss + F.mse_loss(ori_opt,options) + penal_sim
        entropies = entropy(self.Z.codebook, options, self.Z.project_in(option_preds))
        return options, indices, commitment_loss*commitment_loss_weight, entropies

    def get_option(self, word_embeddings, states, timesteps=None, **kwargs):
        # if isinstance(self.state_dim, tuple):
        #     if len(states.shape) != 6: #B,T,N,H,W,C
        #       states = states.reshape(1, -1, *self.state_dim)
        # else:
        #     states = states.reshape(1, -1, self.state_dim)

        timesteps = timesteps.reshape(1, -1)

        attention_mask = torch.ones(1, states.shape[1]).to(device=states.device)

        # max_length = self.option_dt.max_length
        # if max_length is not None:
        #     states = states[:, -max_length:]
        #     timesteps = timesteps[:, -max_length:]
        #
        #     # pad all tokens to sequence length
        #     attention_mask = pad(
        #         torch.ones(1, states.shape[1]),
        #         max_length).to(
        #         dtype=torch.long, device=states.device).reshape(
        #         1, -1)
        #     states = pad(states, max_length).to(dtype=torch.float32)
        #     timesteps = pad(timesteps, max_length).to(dtype=torch.long)
        # else:
        #     attention_mask = None
        #     raise ValueError('Attention mask should not be none')


        options, option_indx, _, _ = self.forward(
            word_embeddings, states, timesteps=timesteps, attention_mask=attention_mask, **kwargs)
        idx= torch.mode(option_indx[0,-states.shape[2]:])[0]
        code = self.Z.codebook[idx]
        option = self.Z.project_out(code)
        #need update, using the last option is incorrect, since the second dimension is T*N
        return option, idx


'''
This is the policy network in LISA
'''
class TrajectoryModel(nn.Module):

    def __init__(self, state_dim, act_dim):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim

    def forward(self, states, actions, rewards, masks=None, attention_mask=None):
        # "masked" tokens or unspecified inputs can be passed in as None
        return None, None, None

    def get_action(self, states, actions, rewards, **kwargs):
        # these will come as tensors on the correct device
        return torch.zeros_like(actions[-1])


# class DecisionTransformer(TrajectoryModel):

#     """
#     This model uses GPT to model (Lang, state_1, state_2, ...) or (state_1, option_1, ...)
#     """

#     def __init__(
#             self,
#             state_dim,
#             action_dim,
#             option_dim,
#             lang_dim,
#             discrete,
#             hidden_size,
#             use_language=False,
#             use_options=True,
#             option_il=False,
#             predict_q=False,
#             max_ep_len=4096,
#             action_tanh=False,
#             no_states=False,
#             no_actions=False,
#             ** kwargs):
#         super().__init__(state_dim, action_dim)

#         self.use_options = use_options
#         self.use_language = use_language
#         self.option_il = option_il
#         self.predict_q = predict_q

#         if use_language and use_options:
#             raise ValueError("Cannot use language and options!")
#         if not use_language and not use_options:
#             raise ValueError("Have to use language or options!")
#         self.option_dim = option_dim
#         self.discrete = discrete

#         self.hidden_size = hidden_size
#         config = transformers.GPT2Config(
#             vocab_size=1,  # doesn't matter -- we don't use the vocab
#             n_embd=hidden_size,
#             **kwargs
#         )

#         self.transformer = GPT2Model(config)

#         #transform timestep to vector embedding
#         self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)

#         self.embed_action = nn.Linear(self.act_dim, hidden_size)

#         self.no_states = no_states
#         self.no_actions = no_actions

#         if use_options:
#             self.embed_option = nn.Linear(self.option_dim, hidden_size)

#         if use_language:
#             self.embed_lang = nn.Linear(lang_dim, hidden_size)

#         self.embed_ln = nn.LayerNorm(hidden_size)
#         # note: we don't predict states or returns for the paper
#         if isinstance(self.state_dim, int):
#             self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
#         self.predict_action = nn.Sequential(
#             *([nn.Linear(hidden_size*2, self.act_dim)] + ([nn.Tanh()] if action_tanh and not discrete else []))
#         )
#         if use_options:
#             self.predict_option = torch.nn.Linear(hidden_size, self.option_dim)
#         if predict_q:
#             self.predict_q = torch.nn.Linear(hidden_size, self.act_dim)

#     def forward(self, state_embeddings, timesteps, options=None, word_embeddings=None, attention_mask=None):
#         # option shape B T*N hidden
#         # state shape B T N H W C
#         # action shape B T act_dim
#         # timesteps/attention_mask B T
#         #batch_size, seq_length,num_cam,H,W,C = states.shape
#         batch_size,seq_length,num_cam,_ = state_embeddings.shape

#         if attention_mask is None:
#             raise ValueError('Should not have attention_mask NONE')
#             # attention mask for GPT: 1 if can be attended to, 0 if not
#             attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

#         if self.use_options:
#             assert options is not None
#             option_embeddings = self.embed_option(options)
#             time_embeddings = self.embed_timestep(timesteps.repeat([1,num_cam]))
#             if option_embeddings.shape[1]<time_embeddings.shape[1] and time_embeddings.shape[1]%option_embeddings.shape[1]==0:
#               factor = time_embeddings.shape[1]/option_embeddings.shape[1]
#               option_embeddings = self.embed_option(options.repeat([1,int(factor),1]))

#             # time embeddings are treated similar to positional embeddings
#             option_embeddings = option_embeddings + time_embeddings

#             if self.no_states:
#                 # IMP: MAKE SURE THIS IS NOT SET ON BY DEFAULT
#                 # state_embeddings = self.embed_state(torch.zeros_like(states))
#                 state_embeddings = torch.zeros_like(state_embeddings)
#             else:
#                 # state_embeddings = self.embed_state(states)
#                 state_embeddings = einops.rearrange(state_embeddings,'b t n c -> b (t n) c')
#                 state_embeddings = state_embeddings + time_embeddings
#                 # state_embeddings = state_embeddings + einops.rearrange(time_embeddings, 'b t c -> b t 1 c') + \
#                 #  einops.rearrange(cam_embeddings, 'n c -> 1 1 n c')

#             # this makes the sequence look like (o1, s1, a1,o2, s2, a2, ...)
#             # which works nice in an autoregressive sense since states predict actions
#             # note that o1 and o2 need not be different
#             # Bryan: it seems the code is equals stack on dim = 2?

#             assert state_embeddings.shape[1]==option_embeddings.shape[1]

#             stacked_inputs = torch.stack(
#                 (option_embeddings, state_embeddings),
#                 dim=1).permute(
#                 0, 2, 1, 3).reshape(
#                 batch_size, 2 * seq_length*num_cam, self.hidden_size)
#             # LAYERNORM
#             stacked_inputs = self.embed_ln(stacked_inputs)

#             # to make the attention mask fit the stacked inputs, have to stack it as well
#             stacked_attention_mask = torch.stack(
#                 (attention_mask.repeat([1,num_cam]), attention_mask.repeat([1,num_cam])), dim=1
#             ).permute(0, 2, 1).reshape(batch_size, 2 * seq_length*num_cam)

#             # we feed in the input embeddings (not word indices as in NLP) to the model
#             transformer_outputs = self.transformer(
#                 inputs_embeds=stacked_inputs,
#                 attention_mask=stacked_attention_mask,
#             )
#             x = transformer_outputs['last_hidden_state']

#             # reshape x so that the second dimension corresponds to the original
#             # options (0), states (1) or actions (2); i.e. x[:,0,t] is the token for s_t
#             traj_out = x.reshape(batch_size, seq_length*num_cam, 2, self.hidden_size).permute(0, 2, 1, 3)
#             # get predictions
#             # predict next action given state and option
#             # in fact, i think only the state info is used for prediction
#             option_state = traj_out[:, :]
#             option_state = einops.rearrange(option_state,'b os nt c -> b nt (os c)') # B T*N 2*hidden
#             action_preds = self.predict_action(option_state) # B T*N act_dim

#             action_preds = action_preds.reshape([batch_size,seq_length,num_cam,-1]) # B T N act_dim

#             action_preds = torch.mean(action_preds,dim=2) # B T act_dim

#             return action_preds
#         else:
#             return None


#     def get_action(self, states, timesteps, options=None, word_embeddings=None, **kwargs):
#         assert options is not None
#         # if isinstance(self.state_dim, tuple):
#         #     if len(states.shape) != 6:  #B,T,N,H,W,C
#         #       states = states.reshape(1, -1, *self.state_dim)
#         # else:
#         #     states = states.reshape(1, -1, self.state_dim)
#         options = options.reshape(1, -1, self.option_dim)
#         timesteps = timesteps.reshape(1, -1)
#         attention_mask = torch.ones(1, states.shape[1]).to(dtype=torch.long, device=states.device).reshape(1, -1)

#         # if max_length is not None:
#         #     states = states[:, -max_length:]
#         #     options = options[:, -max_length:]
#         #     actions = actions[:, -max_length:]
#         #     timesteps = timesteps[:, -max_length:]

#         #     # pad all tokens to sequence length
#         #     # mask future observation
#         #     attention_mask = pad(torch.ones(1, states.shape[1]), max_length)
#         #     states = pad(states, max_length).to(dtype=torch.float32)
#         #     options = pad(options, max_length).to(dtype=torch.float32)
#         #     actions = pad(actions, max_length).to(dtype=torch.float32)
#         #     timesteps = pad(timesteps, max_length).to(dtype=torch.long)
#         # else:
#         #     raise ValueError('Should not have max_length NONE')
#         #     attention_mask = None
#         print("state_embeddings shape: ",states.shape)
#         print("options shape: ",options.shape)
#         print("timesteps shape: ", timesteps.shape)
#         print("attention_mask shape: ",attention_mask.shape)
#         preds = self.forward(
#             states, timesteps, options=options, attention_mask=attention_mask)


#         return preds

class ActionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Lang, state_1, state_2, ...) or (state_1, option_1, ...)
    """

    def __init__(
            self,
            state_dim,
            action_dim,
            option_dim,
            lang_dim,
            discrete,
            hidden_size,
            use_language=False,
            use_options=True,
            option_il=False,
            predict_q=False,
            max_ep_len=4096,
            action_tanh=False,
            no_states=False,
            no_actions=False,
            num_cams = 4,
            ** kwargs):
        super().__init__(state_dim, action_dim)

        self.use_options = use_options
        self.option_il = option_il
        self.predict_q = predict_q


        self.option_dim = option_dim
        self.discrete = discrete
        self.num_cams = num_cams

        self.cam_embedding = nn.Embedding(self.num_cams, hidden_size)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        self.transformer = GPT2Model(config)

        #transform timestep to vector embedding
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)

        self.embed_action = nn.Linear(self.act_dim, hidden_size)

        self.embed_option = nn.Linear(self.option_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        self.pcs_encode = Encoder(hidden_size)
        self.predict_coord = nn.Sequential(
            *dense_layer(hidden_size*3*self.num_cams,hidden_size),
            *dense_layer(hidden_size,hidden_size),
            *dense_layer(hidden_size,3, apply_activation=False)
        )
        self.predict_rot = nn.Sequential(
            *dense_layer(hidden_size*2*self.num_cams,hidden_size),
            *dense_layer(hidden_size,hidden_size),
            *dense_layer(hidden_size,4+1, apply_activation=False)
        )

        self.fuse_action = nn.Sequential(
          *dense_layer((3+4+1)*self.num_cams,3+4+1),
          *dense_layer(3+4+1,3+4+1, apply_activation=False)
        )


    def forward(self, state_embeddings, pcs, timesteps, options=None, word_embeddings=None, attention_mask=None):
        # option shape B T*N hidden
        # state shape B T N H W C
        # timesteps/attention_mask B T
        #batch_size, seq_length,num_cam,H,W,C = states.shape
        batch_size,seq_length,num_cam,_ = state_embeddings.shape

        if attention_mask is None:
            raise ValueError('Should not have attention_mask NONE')
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length))

        assert options is not None
        option_embeddings = self.embed_option(options)
        time_embeddings = self.embed_timestep(timesteps.repeat([1,num_cam]))
        if option_embeddings.shape[1]<time_embeddings.shape[1] and time_embeddings.shape[1]%option_embeddings.shape[1]==0:
          factor = time_embeddings.shape[1]/option_embeddings.shape[1]
          option_embeddings = self.embed_option(options.repeat([1,int(factor),1]))

        # time embeddings are treated similar to positional embeddings
        option_embeddings = option_embeddings + time_embeddings


        state_embeddings = einops.rearrange(state_embeddings,'b t n c -> b (t n) c')
        cam_embeddings = self.cam_embedding(torch.arange(self.num_cams).long().to(state_embeddings.device))
        cam_embeddings = einops.rearrange(cam_embeddings, 'n c -> 1 n c').repeat([1,seq_length,1])
        state_embeddings = state_embeddings + time_embeddings + cam_embeddings


        assert state_embeddings.shape[1]==option_embeddings.shape[1]

        stacked_inputs = torch.stack(
            (option_embeddings, state_embeddings),
            dim=1).permute(
            0, 2, 1, 3).reshape(
            batch_size, 2 * seq_length*num_cam, self.hidden_size)
        # LAYERNORM
        # stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask.repeat([1,num_cam]), attention_mask.repeat([1,num_cam])), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 2 * seq_length*num_cam)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # options (0), states (1); i.e. x[:,0,t] is the token for s_t
        traj_out = x.reshape(batch_size, seq_length*num_cam, 2, self.hidden_size).permute(0, 2, 1, 3)
        # get predictions
        # predict next action given state and option
        # in fact, i think only the state info is used for prediction
        option_state = traj_out[:, :]
        # option_state = einops.rearrange(option_state,'b os tn c -> b tn (os c)') # B T*N 2*hidden
        # rot_preds = self.predict_rot(option_state) # B T*N act_dim

        # rot_preds = einops.rearrange(rot_preds,'b (t n) rot_dim -> b t n rot_dim',t=seq_length,n=num_cam)

        # pcs_embed = einops.rearrange(self.pcs_encode(pcs),'b t n c -> b (t n) c')
        # pcs_embed = torch.cat((pcs_embed,option_state),-1)

        # pred_coord = self.predict_coord(pcs_embed)
        # pred_coord = einops.rearrange(pred_coord,'b (t n) cd_dim -> b t n cd_dim',t=seq_length,n=num_cam)

        # action_preds = torch.cat((pred_coord,rot_preds),-1)

        # action_preds = einops.rearrange(action_preds,'b t n c -> b t (n c)')
        # action_preds = self.fuse_action(action_preds)

        option_state = einops.rearrange(option_state,'b os (t n) c -> b t (n os c)',t=seq_length,n=num_cam) # B T N*2*hidden
        rot_preds = self.predict_rot(option_state) # B T act_dim


        pcs_embed = einops.rearrange(self.pcs_encode(pcs),'b t n c -> b t (n c)')
        pcs_embed = torch.cat((pcs_embed,option_state),-1)

        pred_coord = self.predict_coord(pcs_embed)

        action_preds = torch.cat((pred_coord,rot_preds),-1)

        return action_preds



    def get_action(self, states, pcs,timesteps, options=None, word_embeddings=None, **kwargs):
        assert options is not None
        # if isinstance(self.state_dim, tuple):
        #     if len(states.shape) != 6:  #B,T,N,H,W,C
        #       states = states.reshape(1, -1, *self.state_dim)
        # else:
        #     states = states.reshape(1, -1, self.state_dim)
        options = options.reshape(1, -1, self.option_dim)
        timesteps = timesteps.reshape(1, -1)
        attention_mask = torch.ones(1, states.shape[1]).to(device=states.device).reshape(1, -1)

        # if max_length is not None:
        #     states = states[:, -max_length:]
        #     options = options[:, -max_length:]
        #     actions = actions[:, -max_length:]
        #     timesteps = timesteps[:, -max_length:]

        #     # pad all tokens to sequence length
        #     # mask future observation
        #     attention_mask = pad(torch.ones(1, states.shape[1]), max_length)
        #     states = pad(states, max_length).to(dtype=torch.float32)
        #     options = pad(options, max_length).to(dtype=torch.float32)
        #     actions = pad(actions, max_length).to(dtype=torch.float32)
        #     timesteps = pad(timesteps, max_length).to(dtype=torch.long)
        # else:
        #     raise ValueError('Should not have max_length NONE')
        #     attention_mask = None
        print("state_embeddings shape: ",states.shape)
        print("options shape: ",options.shape)
        print("timesteps shape: ", timesteps.shape)
        print("attention_mask shape: ",attention_mask.shape)
        preds = self.forward(
            states, pcs, timesteps, options=options, attention_mask=attention_mask)


        return preds








class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride_size,
        apply_norm=True,
        apply_activation=True,
    ):
        super().__init__()

        padding_size = (
            kernel_size // 2
            if isinstance(kernel_size, int)
            else (kernel_size[0] // 2, kernel_size[1] // 2)
        )

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride_size,
            padding_size,
            padding_mode="replicate",
        )

        if apply_norm:
            self.norm = nn.GroupNorm(1, out_channels, affine=True)

        if apply_activation:
            self.activation = nn.LeakyReLU(0.02)

    def forward(
        self, ft: torch.Tensor
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        out = self.conv(ft)

        if hasattr(self, "norm"):
            out = self.norm(out)

        if hasattr(self, "activation"):
            out = self.activation(out)

        return out


def dense_layer(in_channels, out_channels, apply_activation=True):
    layer: List[nn.Module] = [nn.Linear(in_channels, out_channels)]
    if apply_activation:
        layer += [nn.LeakyReLU(0.02)]
    return layer

'''
Image Encoder
From https://github.com/suraj-nair-1/lorel/blob/main/models.py
'''
class BaseEncoder(nn.Module):
    __constants__ = ['embedding_size']

    def __init__(self):
        super().__init__()


    def preprocess(self, observations):
        """
            Reshape to 4 dimensions so it works for convolutions
            Chunk the time and batch dimensions
        """
        if len(observations.shape) == 5: #B, T, H,W,C
          observations = torch.permute(observations,(0,1,4,2,3))
          B, T, C, H, W = observations.shape
          return observations.reshape(-1, C, H, W).type(torch.float32).contiguous()
        elif len(observations.shape) == 6: #B,T,N=num_camera,H,W
          observations = torch.permute(observations,(0,1,2,5,3,4))
          B, T, camera, C, H, W = observations.shape
          return observations.reshape(-1, C, H, W).type(torch.float32).contiguous()
        else:
          H,W,C = observations.shape[-3],observations.shape[-2],observations.shape[-1]
          return observations.reshape(-1, C, H, W).type(torch.float32).contiguous()

          
    def unpreprocess(self, embeddings, B, T,N):
        """
            Reshape back to 5 dimensions 
            Unsqueeze the batch and time dimensions
        """
        BTN, E = embeddings.shape
        return embeddings.reshape(B, T, N, E)
        
class Encoder(BaseEncoder):
    __constants__ = ['embedding_size']

    def __init__(self, hidden_size, activation_function='relu', ch=3, robot=False):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(ch, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1) 
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1) 

        self.fc1 = nn.Linear(128*16*16, 1024)
        self.fc2 = nn.Linear(1024,256)
        self.fc3 = nn.Linear(256, hidden_size)

    def forward(self, observations):
        batch,times,n_cam = observations.shape[0],observations.shape[1],observations.shape[2]
        if len(observations.shape) > 4:
            prepro=True
            observations = self.preprocess(observations)
        else:
            prepro = False
            observations = observations
        hidden = self.act_fn(self.conv1(observations))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = hidden.reshape(observations.shape[0], -1)

        hidden = self.act_fn(self.fc1(hidden))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.fc3(hidden)
        
        if prepro:
            del observations
            return self.unpreprocess(hidden, batch, times, n_cam)
        else:
            return hidden

class TwoWayEncoder(BaseEncoder):

    def __init__(self, hidden_size, activation_function='relu', ch=3, robot=False):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv1_1 = nn.Conv2d(ch, 32, 4, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(ch, 32, 4, stride=2, padding=1)
        self.fuse = nn.Conv2d(64, 32, 1, stride=1)      #32*64*64
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1) 
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1) 

        self.fc1 = nn.Linear(128*16*16, 1024)
        self.fc2 = nn.Linear(1024,256)
        self.fc3 = nn.Linear(256, hidden_size-4)

    def forward(self, observations,point_clouds,robot_state):
        batch,times,n_cam = observations.shape[0],observations.shape[1],observations.shape[2]
        robot_state = einops.rearrange(robot_state,'b t c -> (b t) c').repeat([n_cam,1])
        if len(observations.shape) > 4:
            prepro=True
            observations = self.preprocess(observations)
            point_clouds = self.preprocess(point_clouds)
        else:
            prepro = False
            observations = observations
            point_clouds = point_clouds
        hidden_1 = self.act_fn(self.conv1_1(observations))
        hidden_2 = self.act_fn(self.conv1_2(point_clouds))
        hidden = torch.cat([hidden_1,hidden_2],1)
        del hidden_1,hidden_2
        hidden = self.act_fn(self.fuse(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = hidden.reshape(observations.shape[0], -1)

        hidden = self.act_fn(self.fc1(hidden))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.fc3(hidden)
        hidden = torch.cat([hidden,robot_state],-1)
        
        if prepro:
            del observations
            return self.unpreprocess(hidden, batch, times, n_cam)
        else:
            return hidden




def act_layer(act):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'lrelu':
        return nn.LeakyReLU(LRELU_SLOPE)
    elif act == 'elu':
        return nn.ELU()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'prelu':
        return nn.PReLU()
    else:
        raise ValueError('%s not recognized.' % act)


def norm_layer2d(norm, channels):
    if norm == 'batch':
        return nn.BatchNorm2d(channels)
    elif norm == 'instance':
        return nn.InstanceNorm2d(channels, affine=True)
    elif norm == 'layer':
        return nn.GroupNorm(1, channels, affine=True)
    elif norm == 'group':
        return nn.GroupNorm(4, channels, affine=True)
    else:
        raise ValueError('%s not recognized.' % norm)


def norm_layer1d(norm, num_channels):
    if norm == 'batch':
        return nn.BatchNorm1d(num_channels)
    elif norm == 'instance':
        return nn.InstanceNorm1d(num_channels, affine=True)
    elif norm == 'layer':
        return nn.LayerNorm(num_channels)
    else:
        raise ValueError('%s not recognized.' % norm)


class FiLMBlock(nn.Module):
    def __init__(self):
        super(FiLMBlock, self).__init__()

    def forward(self, x, gamma, beta):
        beta = beta.view(x.size(0), x.size(1), 1, 1)
        gamma = gamma.view(x.size(0), x.size(1), 1, 1)

        x = gamma * x + beta

        return x


class Conv2DBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes, strides,
                 norm=None, activation=None, padding_mode='replicate'):
        super(Conv2DBlock, self).__init__()
        padding = kernel_sizes // 2 if isinstance(kernel_sizes, int) else (
            kernel_sizes[0] // 2, kernel_sizes[1] // 2)
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_sizes, strides, padding=padding,
            padding_mode=padding_mode)

        if activation is None:
            nn.init.xavier_uniform_(self.conv2d.weight,
                                    gain=nn.init.calculate_gain('linear'))
            nn.init.zeros_(self.conv2d.bias)
        elif activation == 'tanh':
            nn.init.xavier_uniform_(self.conv2d.weight,
                                    gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(self.conv2d.bias)
        elif activation == 'lrelu':
            nn.init.kaiming_uniform_(self.conv2d.weight, a=LRELU_SLOPE,
                                     nonlinearity='leaky_relu')
            nn.init.zeros_(self.conv2d.bias)
        elif activation == 'relu':
            nn.init.kaiming_uniform_(self.conv2d.weight, nonlinearity='relu')
            nn.init.zeros_(self.conv2d.bias)
        else:
            raise ValueError()

        self.activation = None
        self.norm = None
        if norm is not None:
            self.norm = norm_layer2d(norm, out_channels)
        if activation is not None:
            self.activation = act_layer(activation)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.activation(x) if self.activation is not None else x
        return x


class Conv2DFiLMBlock(Conv2DBlock):

    def __init__(self, in_channels, out_channels, kernel_sizes, strides,
                 norm=None, activation=None, padding_mode='replicate'):
        super(Conv2DFiLMBlock, self).__init__(
            in_channels, out_channels, kernel_sizes, strides, norm, activation,
            padding_mode)

        self.film = FiLMBlock()

    def forward(self, x, gamma, beta):
        x = self.conv2d(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.film(x, gamma, beta)
        x = self.activation(x) if self.activation is not None else x
        return x


class Conv3DBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_sizes: Union[int, list]=3, strides=1,
                 norm=None, activation=None, padding_mode='replicate',
                 padding=None):
        super(Conv3DBlock, self).__init__()
        padding = kernel_sizes // 2 if padding is None else padding
        self.conv3d = nn.Conv3d(
            in_channels, out_channels, kernel_sizes, strides, padding=padding,
            padding_mode=padding_mode)

        if activation is None:
            nn.init.xavier_uniform_(self.conv3d.weight,
                                    gain=nn.init.calculate_gain('linear'))
            nn.init.zeros_(self.conv3d.bias)
        elif activation == 'tanh':
            nn.init.xavier_uniform_(self.conv3d.weight,
                                    gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(self.conv3d.bias)
        elif activation == 'lrelu':
            nn.init.kaiming_uniform_(self.conv3d.weight, a=LRELU_SLOPE,
                                     nonlinearity='leaky_relu')
            nn.init.zeros_(self.conv3d.bias)
        elif activation == 'relu':
            nn.init.kaiming_uniform_(self.conv3d.weight, nonlinearity='relu')
            nn.init.zeros_(self.conv3d.bias)
        else:
            raise ValueError()

        self.activation = None
        self.norm = None
        if norm is not None:
            raise NotImplementedError('Norm not implemented.')
        if activation is not None:
            self.activation = act_layer(activation)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv3d(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.activation(x) if self.activation is not None else x
        return x


class ConvTranspose3DBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_sizes: Union[int, list], strides,
                 norm=None, activation=None, padding_mode='zeros',
                 padding=None):
        super(ConvTranspose3DBlock, self).__init__()
        padding = kernel_sizes // 2 if padding is None else padding
        self.conv3d = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_sizes, strides, padding=padding,
            padding_mode=padding_mode)

        if activation is None:
            nn.init.xavier_uniform_(self.conv3d.weight,
                                    gain=nn.init.calculate_gain('linear'))
            nn.init.zeros_(self.conv3d.bias)
        elif activation == 'tanh':
            nn.init.xavier_uniform_(self.conv3d.weight,
                                    gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(self.conv3d.bias)
        elif activation == 'lrelu':
            nn.init.kaiming_uniform_(self.conv3d.weight, a=LRELU_SLOPE,
                                     nonlinearity='leaky_relu')
            nn.init.zeros_(self.conv3d.bias)
        elif activation == 'relu':
            nn.init.kaiming_uniform_(self.conv3d.weight, nonlinearity='relu')
            nn.init.zeros_(self.conv3d.bias)
        else:
            raise ValueError()

        self.activation = None
        self.norm = None
        if norm is not None:
            self.norm = norm_layer3d(norm, out_channels)
        if activation is not None:
            self.activation = act_layer(activation)

    def forward(self, x):
        x = self.conv3d(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.activation(x) if self.activation is not None else x
        return x


class Conv2DUpsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes, strides,
                 norm=None, activation=None):
        super(Conv2DUpsampleBlock, self).__init__()
        layer = [Conv2DBlock(
            in_channels, out_channels, kernel_sizes, 1, norm, activation)]
        if strides > 1:
            layer.append(nn.Upsample(
                scale_factor=strides, mode='bilinear',
                align_corners=False))
        convt_block = Conv2DBlock(
            out_channels, out_channels, kernel_sizes, 1, norm, activation)
        layer.append(convt_block)
        self.conv_up = nn.Sequential(*layer)

    def forward(self, x):
        return self.conv_up(x)


class Conv3DUpsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, strides, kernel_sizes=3,
                 norm=None, activation=None):
        super(Conv3DUpsampleBlock, self).__init__()
        layer = [Conv3DBlock(
            in_channels, out_channels, kernel_sizes, 1, norm, activation)]
        if strides > 1:
            layer.append(nn.Upsample(
                scale_factor=strides, mode='trilinear',
                align_corners=False))
        convt_block = Conv3DBlock(
            out_channels, out_channels, kernel_sizes, 1, norm, activation)
        layer.append(convt_block)
        self.conv_up = nn.Sequential(*layer)

    def forward(self, x):
        return self.conv_up(x)


class DenseBlock(nn.Module):

    def __init__(self, in_features, out_features, norm=None, activation=None):
        super(DenseBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

        if activation is None:
            nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('linear'))
            nn.init.zeros_(self.linear.bias)
        elif activation == 'tanh':
            nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(self.linear.bias)
        elif activation == 'lrelu':
            nn.init.kaiming_uniform_(self.linear.weight, a=LRELU_SLOPE, nonlinearity='leaky_relu')
            nn.init.zeros_(self.linear.bias)
        elif activation == 'relu':
            nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
            nn.init.zeros_(self.linear.bias)
        else:
            raise ValueError()

        self.activation = None
        self.norm = None
        if norm is not None:
            self.norm = norm_layer1d(norm, out_features)
        if activation is not None:
            self.activation = act_layer(activation)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.activation(x) if self.activation is not None else x
        return x


class SiameseNet(nn.Module):

    def __init__(self,
                 input_channels: List[int],
                 filters: List[int],
                 kernel_sizes: List[int],
                 strides: List[int],
                 norm: str = None,
                 activation: str = 'relu'):
        super(SiameseNet, self).__init__()
        self._input_channels = input_channels
        self._filters = filters
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._norm = norm
        self._activation = activation
        self.output_channels = filters[-1] #* len(input_channels)

    def build(self):
        self._siamese_blocks = nn.ModuleList()
        for i, ch in enumerate(self._input_channels):
            blocks = []
            for i, (filt, ksize, stride) in enumerate(
                    zip(self._filters, self._kernel_sizes, self._strides)):
                conv_block = Conv2DBlock(
                    ch, filt, ksize, stride, self._norm, self._activation)
                blocks.append(conv_block)
            self._siamese_blocks.append(nn.Sequential(*blocks))
        self._fuse = Conv2DBlock(self._filters[-1] * len(self._siamese_blocks),
                                 self._filters[-1], 1, 1, self._norm,
                                 self._activation)

    def forward(self, x):
        if len(x) != len(self._siamese_blocks):
            raise ValueError('Expected a list of tensors of size %d.' % len(
                self._siamese_blocks))
        self.streams = [stream(y) for y, stream in zip(x, self._siamese_blocks)]
        y = self._fuse(torch.cat(self.streams, 1))
        return y


class CNNAndFcsNet(nn.Module):

    def __init__(self,
                 siamese_net: SiameseNet,
                 low_dim_state_len: int,
                 input_resolution: List[int],
                 filters: List[int],
                 kernel_sizes: List[int],
                 strides: List[int],
                 norm: str = None,
                 fc_layers: List[int] = None,
                 activation: str = 'relu'):
        super(CNNAndFcsNet, self).__init__()
        self._siamese_net = copy.deepcopy(siamese_net)
        self._input_channels = self._siamese_net.output_channels + low_dim_state_len
        self._filters = filters
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._norm = norm
        self._activation = activation
        self._fc_layers = [] if fc_layers is None else fc_layers
        self._input_resolution = input_resolution

    def build(self):
        self._siamese_net.build()
        layers = []
        channels = self._input_channels
        for i, (filt, ksize, stride) in enumerate(
                list(zip(self._filters, self._kernel_sizes, self._strides))[
                :-1]):
            layers.append(Conv2DBlock(
                channels, filt, ksize, stride, self._norm, self._activation))
            channels = filt
        layers.append(Conv2DBlock(
            channels, self._filters[-1], self._kernel_sizes[-1],
            self._strides[-1]))
        self._cnn = nn.Sequential(*layers)
        self._maxp = nn.AdaptiveMaxPool2d(1)

        channels = self._filters[-1]
        dense_layers = []
        for n in self._fc_layers[:-1]:
            dense_layers.append(
                DenseBlock(channels, n, activation=self._activation))
            channels = n
        dense_layers.append(
            DenseBlock(channels, self._fc_layers[-1]))
        self._fcs = nn.Sequential(*dense_layers)

    def forward(self, observations, low_dim_ins):
        x = self._siamese_net(observations)
        _, _, h, w = x.shape
        low_dim_latents = low_dim_ins.unsqueeze(
            -1).unsqueeze(-1).repeat(1, 1, h, w)
        combined = torch.cat([x, low_dim_latents], dim=1)
        x = self._cnn(combined)
        x = self._maxp(x).squeeze(-1).squeeze(-1)
        return self._fcs(x)


class CNNLangAndFcsNet(nn.Module):

    def __init__(self,
                 siamese_net: SiameseNet,
                 low_dim_state_len: int,
                 input_resolution: List[int],
                 filters: List[int],
                 kernel_sizes: List[int],
                 strides: List[int],
                 norm: str = None,
                 fc_layers: List[int] = None,
                 activation: str = 'relu'):
        super(CNNLangAndFcsNet, self).__init__()
        self._siamese_net = copy.deepcopy(siamese_net)
        self._input_channels = self._siamese_net.output_channels + low_dim_state_len
        self._filters = filters
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._norm = norm
        self._activation = activation
        self._fc_layers = [] if fc_layers is None else fc_layers
        self._input_resolution = input_resolution

        self._lang_feat_dim = 1024

    def build(self):
        self._siamese_net.build()
        layers = []
        channels = self._input_channels

        self.conv1 = Conv2DFiLMBlock(
            channels, self._filters[0], self._kernel_sizes[0],
            self._strides[0])
        self.gamma1 = nn.Linear(self._lang_feat_dim, self._filters[0])
        self.beta1 = nn.Linear(self._lang_feat_dim, self._filters[0])

        self.conv2 = Conv2DFiLMBlock(
            self._filters[0], self._filters[1], self._kernel_sizes[1],
            self._strides[1])
        self.gamma2 = nn.Linear(self._lang_feat_dim, self._filters[1])
        self.beta2 = nn.Linear(self._lang_feat_dim, self._filters[1])

        self.conv3 = Conv2DFiLMBlock(
            self._filters[1], self._filters[2], self._kernel_sizes[2],
            self._strides[2])
        self.gamma3 = nn.Linear(self._lang_feat_dim, self._filters[2])
        self.beta3 = nn.Linear(self._lang_feat_dim, self._filters[2])

        self._maxp = nn.AdaptiveMaxPool2d(1)

        channels = self._filters[-1]
        dense_layers = []
        for n in self._fc_layers[:-1]:
            dense_layers.append(
                DenseBlock(channels, n, activation=self._activation))
            channels = n
        dense_layers.append(
            DenseBlock(channels, self._fc_layers[-1]))
        self._fcs = nn.Sequential(*dense_layers)

    def forward(self, observations, low_dim_ins, lang_goal_emb):
        x = self._siamese_net(observations)
        _, _, h, w = x.shape
        low_dim_latents = low_dim_ins.unsqueeze(
            -1).unsqueeze(-1).repeat(1, 1, h, w)
        combined = torch.cat([x, low_dim_latents], dim=1)

        g1 = self.gamma1(lang_goal_emb)
        b1 = self.beta1(lang_goal_emb)
        x = self.conv1(combined, g1, b1)

        g2 = self.gamma2(lang_goal_emb)
        b2 = self.beta2(lang_goal_emb)
        x = self.conv2(x, g2, b2)

        g3 = self.gamma3(lang_goal_emb)
        b3 = self.beta3(lang_goal_emb)
        x = self.conv3(x, g3, b3)

        x = self._maxp(x).squeeze(-1).squeeze(-1)
        return self._fcs(x)


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# ViT IO implementation adpated for baseline
# Source: https://github.com/lucidrains/vit-pytorch
# License: https://github.com/lucidrains/vit-pytorch/blob/main/LICENSE

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        self.num_patches_x = (image_height // patch_height)
        self.num_patches_y = (image_width // patch_width)
        self.num_patches = self.num_patches_x * self.num_patches_y
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = einops.repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = x[:, 1:].reshape(b, -1, self.num_patches_x, self.num_patches_y)

        return x


class ViTLangAndFcsNet(nn.Module):

    def __init__(self,
                 vit: ViT,
                 low_dim_state_len: int,
                 input_resolution: List[int],
                 filters: List[int],
                 kernel_sizes: List[int],
                 strides: List[int],
                 norm: str = None,
                 fc_layers: List[int] = None,
                 activation: str = 'relu'):
        super(ViTLangAndFcsNet, self).__init__()
        self._vit = copy.deepcopy(vit)
        self._input_channels = 64 + low_dim_state_len
        self._filters = filters
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._norm = norm
        self._activation = activation
        self._fc_layers = [] if fc_layers is None else fc_layers
        self._input_resolution = input_resolution

        self._lang_feat_dim = 1024

    def build(self):
        layers = []
        channels = self._input_channels

        self.conv1 = Conv2DFiLMBlock(
            channels, self._filters[0], self._kernel_sizes[0],
            self._strides[0])
        self.gamma1 = nn.Linear(self._lang_feat_dim, self._filters[0])
        self.beta1 = nn.Linear(self._lang_feat_dim, self._filters[0])

        self.conv2 = Conv2DFiLMBlock(
            self._filters[0], self._filters[1], self._kernel_sizes[1],
            self._strides[1])
        self.gamma2 = nn.Linear(self._lang_feat_dim, self._filters[1])
        self.beta2 = nn.Linear(self._lang_feat_dim, self._filters[1])

        self.conv3 = Conv2DFiLMBlock(
            self._filters[1], self._filters[2], self._kernel_sizes[2],
            self._strides[2])
        self.gamma3 = nn.Linear(self._lang_feat_dim, self._filters[2])
        self.beta3 = nn.Linear(self._lang_feat_dim, self._filters[2])

        self._maxp = nn.AdaptiveMaxPool2d(1)

        channels = self._filters[-1]
        dense_layers = []
        for n in self._fc_layers[:-1]:
            dense_layers.append(
                DenseBlock(channels, n, activation=self._activation))
            channels = n
        dense_layers.append(
            DenseBlock(channels, self._fc_layers[-1]))
        self._fcs = nn.Sequential(*dense_layers)

    def forward(self, observations, low_dim_ins, lang_goal_emb):
        rgb_depth = torch.cat([*observations], dim=1)
        x = self._vit(rgb_depth)
        _, _, h, w = x.shape
        low_dim_latents = low_dim_ins.unsqueeze(
            -1).unsqueeze(-1).repeat(1, 1, h, w)
        combined = torch.cat([x, low_dim_latents], dim=1)

        g1 = self.gamma1(lang_goal_emb)
        b1 = self.beta1(lang_goal_emb)
        x = self.conv1(combined, g1, b1)

        g2 = self.gamma2(lang_goal_emb)
        b2 = self.beta2(lang_goal_emb)
        x = self.conv2(x, g2, b2)

        g3 = self.gamma3(lang_goal_emb)
        b3 = self.beta3(lang_goal_emb)
        x = self.conv3(x, g3, b3)

        x = self._maxp(x).squeeze(-1).squeeze(-1)
        return self._fcs(x)



class Conv3DInceptionBlockUpsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor,
                 norm=None, activation=None, residual=False):
        super(Conv3DInceptionBlockUpsampleBlock, self).__init__()
        layer = []

        convt_block = Conv3DInceptionBlock(
            in_channels, out_channels, norm, activation)
        layer.append(convt_block)

        if scale_factor > 1:
            layer.append(nn.Upsample(
                scale_factor=scale_factor, mode='trilinear',
                align_corners=False))

        convt_block = Conv3DInceptionBlock(
            out_channels, out_channels, norm, activation)
        layer.append(convt_block)

        self.conv_up = nn.Sequential(*layer)

    def forward(self, x):
        return self.conv_up(x)


class Conv3DInceptionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, norm=None, activation=None,
                 residual=False):
        super(Conv3DInceptionBlock, self).__init__()
        self._residual = residual
        cs = out_channels // 4
        assert out_channels % 4 == 0
        latent = 32
        self._1x1conv = Conv3DBlock(
            in_channels, cs * 2, kernel_sizes=1, strides=1, norm=norm,
            activation=activation)

        self._1x1conv_a = Conv3DBlock(
            in_channels, latent, kernel_sizes=1, strides=1, norm=norm,
            activation=activation)
        self._3x3conv = Conv3DBlock(
            latent, cs, kernel_sizes=3, strides=1,
            norm=norm, activation=activation)

        self._1x1conv_b = Conv3DBlock(
            in_channels, latent, kernel_sizes=1, strides=1, norm=norm,
            activation=activation)
        self._5x5_via_3x3conv_a = Conv3DBlock(
            latent, latent, kernel_sizes=3, strides=1, norm=norm,
            activation=activation)
        self._5x5_via_3x3conv_b = Conv3DBlock(
            latent, cs, kernel_sizes=3, strides=1, norm=norm,
            activation=activation)
        self.out_channels = out_channels + (in_channels if residual else 0)

    def forward(self, x):
        yy = []
        if self._residual:
            yy = [x]
        return torch.cat(yy + [self._1x1conv(x),
                               self._3x3conv(self._1x1conv_a(x)),
                               self._5x5_via_3x3conv_b(self._5x5_via_3x3conv_a(
                                   self._1x1conv_b(x)))], 1)

class ConvTransposeUp3DBlock(nn.Module):

    def __init__(self, in_channels, out_channels, strides=2, padding=0,
                 norm=None, activation=None, residual=False):
        super(ConvTransposeUp3DBlock, self).__init__()
        self._residual = residual

        self._1x1conv = Conv3DBlock(
            in_channels, out_channels, kernel_sizes=1, strides=1, norm=norm,
            activation=activation)
        self._3x3conv = ConvTranspose3DBlock(
            out_channels, out_channels, kernel_sizes=2, strides=strides, norm=norm,
            activation=activation, padding=padding)
        self._1x1conv_a = Conv3DBlock(
            out_channels, out_channels, kernel_sizes=1, strides=1, norm=norm,
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self._1x1conv(x)
        x = self._3x3conv(x)
        x = self._1x1conv_a(x)
        return x


class SpatialSoftmax3D(torch.nn.Module):

    def __init__(self, depth, height, width, channel):
        super(SpatialSoftmax3D, self).__init__()
        self.depth = depth
        self.height = height
        self.width = width
        self.channel = channel
        self.temperature = 0.01
        pos_x, pos_y, pos_z = np.meshgrid(
            np.linspace(-1., 1., self.depth),
            np.linspace(-1., 1., self.height),
            np.linspace(-1., 1., self.width)
        )
        pos_x = torch.from_numpy(
            pos_x.reshape(self.depth * self.height * self.width)).float()
        pos_y = torch.from_numpy(
            pos_y.reshape(self.depth * self.height * self.width)).float()
        pos_z = torch.from_numpy(
            pos_z.reshape(self.depth * self.height * self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)
        self.register_buffer('pos_z', pos_z)

    def forward(self, feature):
        feature = feature.view(
            -1, self.height * self.width * self.depth)  # (B, c*d*h*w)
        softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=1,
                               keepdim=True)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=1,
                               keepdim=True)
        expected_z = torch.sum(self.pos_z * softmax_attention, dim=1,
                               keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y, expected_z], 1)
        feature_keypoints = expected_xy.view(-1, self.channel * 3)
        return feature_keypoints