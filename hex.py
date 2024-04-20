import copy
import torch
import torch.nn as nn
from transformers import DistilBertModel
from basemodel import *
from hive.transformer_unet import TransformerUNet
from peract.qattention_peract_bc_agent import QAttentionPerActBCAgent
from peract.perceiver_lang_io import PerceiverVoxelLangEncoder
from lamb import Lamb

from utils import *

class Hex(nn.Module):

  def __init__(self, device, args,max_length):
    super().__init__()

    self.device = device
    self.eval_option_fre = 1

    self.state_dim = args.env.state_dim
    self.action_dim = args.env.action_dim
    self.option_dim = args.option_dim

    encoder_args = {'hidden_size': args.dt.hidden_size,'ch': 3}
    self.encoder = TwoWayEncoder(**encoder_args)
  
    args.option_selector.option_transformer.max_length = int(max_length)
    args.option_selector.option_transformer.max_ep_len = int(max_length)
    option_selector_args = dict(args.option_selector)
    option_selector_args['state_dim'] = args.env.state_dim
    option_selector_args['option_dim'] = args.option_dim
    option_selector_args['codebook_dim'] = args.codebook_dim
    option_selector_args['num_cams'] = args.env.num_cams
    self.option_selector = OptionSelector(lang_dim=args.env.lang_feat_dim, **option_selector_args)

    perceiver_encoder = PerceiverVoxelLangEncoder(
      depth=args.peract.transformer_depth,
      iterations=args.peract.transformer_iterations,
      voxel_size=args.peract.voxel_sizes[0],
      initial_dim = 3 + 3 + 1 + 3,
      low_dim_size=4,
      layer=0,
      num_rotation_classes=int(360. // args.peract.rotation_resolution),
      num_grip_classes=2,
      num_collision_classes=2,
      input_axis=3,
      num_latents = args.peract.num_latents,
      latent_dim = args.peract.latent_dim,
      cross_heads = args.peract.cross_heads,
      latent_heads = args.peract.latent_heads,
      cross_dim_head = args.peract.cross_dim_head,
      latent_dim_head = args.peract.latent_dim_head,
      weight_tie_layers = False,
      activation = args.peract.activation,
      pos_encoding_with_lang=args.peract.pos_encoding_with_lang,
      input_dropout=args.peract.input_dropout,
      attn_dropout=args.peract.attn_dropout,
      decoder_dropout=args.peract.decoder_dropout,
      lang_fusion_type=args.peract.lang_fusion_type,
      voxel_patch_size=args.peract.voxel_patch_size,
      voxel_patch_stride=args.peract.voxel_patch_stride,
      no_skip_connection=args.peract.no_skip_connection,
      no_perceiver=args.peract.no_perceiver,
      no_language=args.peract.no_language,
      final_dim=args.peract.final_dim,
    )

    self.act_transformer = QAttentionPerActBCAgent(
      layer=0,
      coordinate_bounds=args.env.scene_bounds,
      perceiver_encoder=perceiver_encoder,
      camera_names=args.env.cameras,
      voxel_size=args.peract.voxel_sizes[0],
      bounds_offset=None,
      image_crop_size=args.peract.image_crop_size,
      lr=args.peract.lr,
      training_iterations=args.total_steps,
      lr_scheduler=args.peract.lr_scheduler,
      num_warmup_steps=args.peract.num_warmup_steps,
      trans_loss_weight=args.peract.trans_loss_weight,
      rot_loss_weight=args.peract.rot_loss_weight,
      grip_loss_weight=args.peract.grip_loss_weight,
      collision_loss_weight=args.peract.collision_loss_weight,
      include_low_dim_state=True,
      image_resolution=args.env.camera_resolution,
      batch_size=1,      #
      voxel_feature_size=3,
      lambda_weight_l2=args.peract.lambda_weight_l2,
      num_rotation_classes=int(360. // args.peract.rotation_resolution),
      rotation_resolution=args.peract.rotation_resolution,
      transform_augmentation=args.peract.transform_augmentation.apply_se3,
      transform_augmentation_xyz=args.peract.transform_augmentation.aug_xyz,
      transform_augmentation_rpy=args.peract.transform_augmentation.aug_rpy,
      transform_augmentation_rot_resolution=args.peract.transform_augmentation.aug_rot_resolution,
      optimizer_type=args.peract.optimizer,
      num_devices=1,
    )
    self.act_transformer.build(training=True, device=self.device)

    self.loss_fn = nn.CrossEntropyLoss(reduction='none')
    params = list(self.option_selector.parameters())+list(self.act_transformer._q.parameters())

    self.optimizer = Lamb(
                    params,
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay,
                    betas=(0.9, 0.999),
                    adam=False,
                )


    self.scheduler = torch.optim.lr_scheduler.LinearLR(
      self.optimizer,start_factor=1.0,end_factor=0.5,total_iters=3000)
 
  def compute_loss(self,pred,labels):
    pred = pred.view(-1,pred.size(-1))
    labels = labels.view(-1,labels.size(-1)).argmax(-1)
    return self.loss_fn(pred,labels)

  def forward(self,steps, lang_goal_emb, lang_token_embs, obs ,pcs, robot_state, actions, trans_idx, rot_idx,  timesteps, attention_mask=None,min_interval=None):
    # obs: B, T, N, H, W, C
    # pcs: B, T, N, H, W, C
    # actions: B, T, act_dim
    # timesteps: B, T
    # states = torch.cat([states,pc_obs],dim=2)
    obs = (obs.float() / 255.0) * 2.0 - 1.0
    state_embedding = self.encoder(obs,pcs,robot_state)

    selected_options, _, commitment_loss, etrp = self.option_selector(lang_goal_emb, state_embedding,
     timesteps=timesteps, attention_mask=attention_mask)
    
    for t in range(obs.shape[1]):

      if (actions[:,t,:7]==0).all() or (pcs[:,t]==0).all():
        continue
      
      ob = []
      pc = []
      for n in range(obs[:,t].shape[1]):
        rgb = einops.rearrange(obs[:,t,n],'b h w c -> b c h w')
        p_c = einops.rearrange(pcs[:,t,n],'b h w c -> b c h w')
        ob.append([rgb,p_c])
        pc.append(p_c)
      proprio = robot_state[:,t]
      t_idx = trans_idx[:,t]
      r_idx = rot_idx[:,t]
      action = actions[:,t,:7]
      
      if t == 0:
        selected_option = selected_options[:,t]
      else:
        selected_option = selected_options[:,t].clone().detach()

      loss = self.act_transformer.update(steps,ob,pc,proprio,t_idx,r_idx,action, lang_goal_emb,lang_token_embs,
        selected_option)
      if t == 0:
        total_loss = loss + commitment_loss.mean()
      else:
        total_loss = loss

      total_loss.backward()
      self.optimizer.step()
      self.optimizer.zero_grad()

    if self.scheduler is not None:
      self.scheduler.step(steps)

    return total_loss



  def _normalize_quat(self, x):
    return x / x.square().sum().sqrt().unsqueeze(-1)

  def get_action(self, lang_goal_emb, lang_token_embs,obs, pcs, robot_state, timesteps, options=None,**kwargs):
    obs = (obs.float() / 255.0) * 2.0 - 1.0
    state_embedding = self.encoder(obs,pcs,robot_state)


    option, option_index = self.option_selector.get_option(word_embeddings=lang_goal_emb,
                                                        states=state_embedding,
                                                        timesteps=timesteps, **kwargs)
    options[-1] = option
    option = option.unsqueeze(0)

    ob = []
    pc = []
    for n in range(obs[:,0].shape[1]):
      rgb = einops.rearrange(obs[:,0,n],'b h w c -> b c h w')
      p_c = einops.rearrange(pcs[:,0,n],'b h w c -> b c h w')
      ob.append([rgb,p_c])
      pc.append(p_c)
    proprio = robot_state[:,0]

    preds = self.act_transformer.act(ob, pc, proprio, lang_goal_emb,lang_token_embs,option)

    action =preds

    ignore_collisions = np.array([1.0])
    action = np.concatenate([action.squeeze(), ignore_collisions])

    out_action = action

    return out_action, option_index

  def save(self, iter_num, filepath, config):
    if hasattr(self.model, 'module'):
      model = self.model.module

    torch.save({'model': model.state_dict(),
      'optimizer': self.optimizer.state_dict(),
      'scheduler': self.scheduler.state_dict(),
      'iter_num': iter_num,
      'train_dataset_max_length': self.option_selector.option_transformer.max_length,
      'config': config}, filepath)

  def load(self, filepath):
    checkpoint = torch.load(filepath)
    self.model.load_state_dict(checkpoint['model'])
    self.optimizer.load_state_dict(checkpoint['optimizer'])
    self.scheduler.load_state_dict(checkpoint['scheduler'])
    return {'iter_num': checkpoint['iter_num'], 'train_dataset_max_length': checkpoint['train_dataset_max_length'], 'config': checkpoint['config']}

