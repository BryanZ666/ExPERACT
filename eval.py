import torch
import numpy as np
import torchvision.transforms.functional as transforms_f
import einops
from utils import extract_obs,create_obs_config
import time
from itertools import chain

from rlbench.utils import get_stored_demos
from rlbench.backend.exceptions import InvalidActionError
from pyrep.errors import IKError, ConfigurationPathError
from clip.core.clip import build_model, load_clip, tokenize
from clip.core.simple_tokenizer import SimpleTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

def eval_task(data_path, num_demos, task, cameras, img_size, episode_length, eval_model, words_dict, sample_fre =10):
  if hasattr(eval_model, 'module'):
    eval_model = eval_model.module
  obs_cfg = create_obs_config(cameras,[img_size,img_size])
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  stacked_opt_list = []
  lm, _ = load_clip('RN50', jit=False, device=device)
  clip_model = build_model(lm.state_dict())
  clip_model.to(device=device)
  del lm

  if hasattr(eval_model,"option_selector"):
    option_dim = eval_model.option_selector.option_dim
  else:
    option_dim = 0

  maximum_len = 0
  for idx in range(num_demos):
    options = torch.zeros((0, option_dim), device=device)
    options_list = []
    demo = get_stored_demos(amount=1, image_paths=False,
      dataset_root=data_path,
      variation_number=-1, task_name=task,
      obs_config=obs_cfg,
      random_selection=False,
      from_episode_number=idx)[0]


    demo_len = len(demo)
    maximum_len = max(maximum_len,demo_len)

    #descs consists of multiple text expression of the task.
    descs = demo._observations[0].misc['descriptions']
    descs = str(np.random.choice(descs))
    print(descs)
    # descs = descs[0]

    for ts in range(demo_len):
      # if ts % sample_fre !=0:
      #   continue
      if ts >= 512:
        break
      init_obs = demo[ts]

      states,pcs,rbs = extract_rgb_ob(init_obs,cameras)
      
      all_tokens = list(descs.split(" "))

      tokens = tokenize([descs]).numpy()
      token_tensor = torch.from_numpy(tokens).to(device=device)
      sentence_emb, token_embs = clip_model.encode_text_with_embeddings(token_tensor)
      lang_goal_emb = sentence_emb[0].unsqueeze(0).float().detach()
      lang_token_embs = token_embs[0].unsqueeze(0).float().detach()

      # B, T, N, H, W, C
      states = einops.rearrange(torch.from_numpy(states),'n h w c -> 1 1 n h w c').to(device=device)
      pcs = einops.rearrange(torch.from_numpy(pcs), 'n h w c -> 1 1 n h w c').to(device=device)
      rbs = einops.rearrange(torch.from_numpy(rbs), 'c -> 1 1 c').to(device=device)

      options = torch.cat([options, torch.zeros((1, option_dim), device=device)], dim=0)
      timesteps = torch.arange(ts,ts+1,device=device).reshape(1, -1)

      action, ops_idx = get_option_action(
        eval_model, states, pcs, rbs, options, timesteps, lang_goal_emb,lang_token_embs)
      options_list.append(ops_idx.cpu().item())
      for w in all_tokens:
        words_dict.setdefault(ops_idx.cpu().item(),[]).append(w)
    stacked_opt_list.append(options_list)

  return words_dict,stacked_opt_list,maximum_len


def eval_multiple_task_demo(data_path, num_demos, task_list, cameras, img_size,episode_length,eval_model,save_note):
  words_dict = {}
  for task in task_list:
    print(task)
    words_dict,stacked_opt_list,max_len = eval_task(data_path, num_demos, task, cameras, img_size,episode_length,eval_model,words_dict)

    matrix = np.zeros([num_demos, max_len])
    for i in range(num_demos):
      for j in range(max_len):
        if j < len(stacked_opt_list[i]):
          matrix[i,j] +=int(stacked_opt_list[i][j])
    
    plt.figure(figsize=(30, 10))
    matrix = (matrix)/(20.0 + 1e-6)
    sns.heatmap(matrix, yticklabels=list(range(num_demos)))
    # plt.plot()
    plt.savefig(f"./viz/{save_note}_{task}_train_demo_options.png")


  viz_matrix(words_dict, 20, f"./viz/{save_note}_training_dataset_")

  return words_dict


def viz_matrix(words_dict, num_options, file_path):
  skip_words = ['go', 'to', 'the', 'a', '[SEP]']
  words = sorted(set(chain(*words_dict.values())) - set(skip_words))

  def w_to_ind(word):
    return words.index(word)

  matrix = np.zeros([len(words), num_options])

  for o in range(num_options):
    if o in words_dict:
      for w in words_dict[o]:
        if w not in skip_words:
          matrix[w_to_ind(w), o] += 1

  # plot co-occurence matrix (words x options)
  plt.figure(figsize=(30, 10))
  sns.heatmap(matrix, yticklabels=words)
  # plt.plot()
  plt.savefig(file_path+"co_occurence_matrix.png")

  # Now if we normalize it by column (word freq for each option)
  plt.figure(figsize=(30, 10))
  matrix_norm_col = (matrix)/(matrix.sum(axis=0, keepdims=True) + 1e-6)
  sns.heatmap(matrix_norm_col, yticklabels=words)
  plt.savefig(file_path+"word_freq.png")

  # Now if we normalize it by row (option freq for each word)
  plt.figure(figsize=(30, 10))
  matrix_norm_row = (matrix)/(matrix.sum(axis=1, keepdims=True) + 1e-6)
  sns.heatmap(matrix_norm_row, yticklabels=words)
  plt.savefig(file_path+"opt_freq.png")
  plt.close()

def extract_rgb_ob(observation,cameras):
  observation = extract_obs(observation)
  ob = []
  pcs =[]
  rbstate = np.array(observation['robot_state'])
  for camera in cameras:
    rgb_name = "%s_%s" % (camera, 'rgb')
    rgb = np.array(observation[rgb_name])
    rgb = rgb/np.max(rgb)*255
    rgb = rgb.astype(np.uint8)
    # rgb = transforms_f.normalize(
    #   torch.from_numpy(rgb).to(torch.float32), 
    #   [0.485, 0.456, 0.406], 
    #   [0.229, 0.224, 0.225]
    # )

    rgb = np.transpose(rgb, (1, 2, 0))

    ob.append(rgb)

    pc_name = "%s_%s" % (camera, 'point_cloud')
    pc = np.array(observation[pc_name])
    pc = np.transpose(pc, (1, 2, 0))
    pcs.append(pc)
  ob =np.stack(ob)
  pcs = np.stack(pcs)
  return ob, pcs,rbstate

# def eval_episode(env, tokenizer, model, max_ep_len, words_dict, render, device,action_target,cameras, **kwargs):
#   """Evaluate a single episode."""
#   images = []
#   options_list = []
  
#   lang, states = env.reset()
#   states,pcs,rbs = extract_rgb_ob(states,cameras) # N, H, W, C
#   lang = lang[0]
#   if render:
#     images.append(states[0].squeeze())

#   state_dim = states.shape
#   act_dim = action_target.shape[-1]
#   if hasattr(model,"option_selector"):
#     option_dim = model.option_selector.option_dim
#   else:
#     option_dim = 0
  
#   # we keep all the histories on the device
#   # note that the latest action and reward will be "padding"
#   with torch.no_grad():
#     lm_input = tokenizer(text=[lang], add_special_tokens=True,
#       return_tensors='pt', padding=True).to(device=device)

#   with torch.no_grad():
#     lm_embeddings = model.lm(
#       lm_input['input_ids'], lm_input['attention_mask']).last_hidden_state
#     cls_embeddings = lm_embeddings[:, 0, :]
#     # word_embeddings = lm_embeddings[:, 1:-1, :]      # skip the CLS and SEP tokens. here there's no padding so this is actually the CLS and SEP
#     word_embeddings = lm_embeddings[:, 1:, :]      # skip the CLS tokens

#   if isinstance(state_dim, tuple):
#      # B, T, N, H, W, C
#     states = einops.rearrange(torch.from_numpy(states),'n h w c -> 1 1 n h w c').to(device=device, dtype=torch.float32)
#     pcs = einops.rearrange(torch.from_numpy(pcs), 'n h w c -> 1 1 n h w c').to(device=device,dtype=torch.float32)
#     rbs = einops.rearrange(torch.from_numpy(rbs), 'c -> 1 1 c').to(device=device,dtype=torch.float32)
#   else:
#     states = states.reshape(1, state_dim).to(device=device, dtype=torch.float32)
#     pcs = pcs.reshape(1, state_dim).to(device=device, dtype=torch.float32)
#     rbs = rbs.reshape(1, -1).to(device=device, dtype=torch.float32)

#   actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
#   options = torch.zeros((0, option_dim), device=device, dtype=torch.float32)

#   episode_return, episode_length = 0, 0
#   state = torch.clone(states)
#   pc = torch.clone(pcs)
#   rb = torch.clone(rbs)

#   for t in range(max_ep_len):
#     # add dummy action
#     actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
#     options = torch.cat([options, torch.zeros((1, option_dim), device=device)], dim=0)
#     timesteps = torch.arange(t,t+1,device=device, dtype=torch.long).reshape(1, -1)
#     action = torch.zeros((1, act_dim), device=device)
#     action, options = get_option_action(
#       model, state, pc, rb, action, options, timesteps, word_embeddings, **kwargs)

#     # action = get_action(model,states,pcs,lm_input['input_ids'],lm_input['attention_mask'],timesteps)

#     actions[-1] = action

#     print("action & actions shape: ",action.shape,"*****"*10, actions.shape)

#     ignore_collisions = torch.Tensor([1.0]).to(states.device)
#     action = torch.cat([action.squeeze(), ignore_collisions])

#     # if model.decision_transformer.discrete:
#     #   actions[-1] = torch.nn.functional.one_hot(action, num_classes=act_dim)
#     # else:
#     #   pass
#     #   #need update
      
#     #   # action = torch.clamp(action, torch.from_numpy(env.action_space.low).to(
#     #   #   device), torch.from_numpy(env.action_space.high).to(device))
#     #   # actions[-1] = action

#     # if t >= len(action_target):
#     #   break
#     # action = action_target[t]
#     # while (action==0).all():
#     #   action_target=torch.cat((action_target[:t],action_target[t+1:]),dim=0)
#     #   action = action_target[t]
#     # ignore_col = torch.Tensor([1.0]).to(device)
#     # action = torch.cat([action, ignore_col])
#     # print("target action: ",action_target[t],"="*20,action)

#     action = action.detach().cpu().numpy()
#     try:
#       state, reward, done = env.step(action)

#       state,pc,rbs = extract_rgb_ob(state,cameras)
#       print("state shape: ", state.shape)
#       if render:
#         images.append(state[0].squeeze())

#       episode_return += reward
#       episode_length += 1
    
#     except (IKError, ConfigurationPathError, InvalidActionError) as e:
#       done = True
#       reward = 0.0
        
#     # if isinstance(state_dim, tuple):
#     #   cur_state = torch.from_numpy(state).to(device=device).reshape(1, *state_dim)
#     # else:
#     #   cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
#     # states = torch.cat([states, cur_state], dim=0).float()
#     # timesteps = torch.cat(
#     #   [timesteps,torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

    

#     if done:
#       break
#     state = einops.rearrange(torch.from_numpy(state),'n h w c -> 1 1 n h w c').to(device=device, dtype=torch.float32)
#     # print("states & state shape: ",states.shape,"@"*20,state.shape)
#     # states = torch.cat([states,state],dim=1)
#     pc = einops.rearrange(torch.from_numpy(pc), 'n h w c -> 1 1 n h w c').to(device=device, dtype=torch.float32)
#     # pcs = torch.cat([pcs, pc], dim=1)
#     rb = einops.rearrange(torch.from_numpy(rbs), 'c -> 1 1 c').to(device=device,dtype=torch.float32)

#   del states,actions,timesteps,options
#   # images = np.stack(images,axis=0)
#   # images = torch.from_numpy(images)*255
#   # images = einops.rearrange(images.to(torch.float32),'t n h w c -> (t n) h w c')
#   # print("image type: ",type(images))
#   # print("images shape: ",images.shape,"       ",torch.max(images))
#   # print("image mean",torch.mean(images))
#   return episode_return, episode_length, options_list, images, words_dict

def eval_episode(env, model, max_ep_len, words_dict, render, device,action_target,cameras,seed, **kwargs):
  """Evaluate a single episode."""
  images = []
  options_list = []

  fail_sig = False
  
  time.sleep(1)
  env.set_variation(-1)
  d = env.get_demos(
      1, live_demos=False, random_selection=False, from_episode_number=seed)[0]

  env.set_variation(d.variation_number)
  lang, states = env.reset_to_demo(d)
  # lang, states = env.reset()
  states,pcs,rbs = extract_rgb_ob(states,cameras) # N, H, W, C

  lang = str(np.random.choice(lang))
  print(lang)
  # lang = lang[0]
  # print(lang)
  if render:
    images.append(states[0].squeeze())

  state_dim = states.shape
  act_dim = action_target.shape[-1]
  if hasattr(model,"option_selector"):
    option_dim = model.option_selector.option_dim
  else:
    option_dim = 0
  
  lm, _ = load_clip('RN50', jit=False, device=device)
  clip_model = build_model(lm.state_dict())
  clip_model.to(device=device)
  del lm
  
  all_tokens = list(lang.split(" "))

  tokens = tokenize([lang]).numpy()
  token_tensor = torch.from_numpy(tokens).to(device=device)
  sentence_emb, token_embs = clip_model.encode_text_with_embeddings(token_tensor)
  lang_goal_emb = sentence_emb[0].unsqueeze(0).float().detach()
  lang_token_embs = token_embs[0].unsqueeze(0).float().detach()



  # B, T, N, H, W, C
  states = einops.rearrange(torch.from_numpy(states),'n h w c -> 1 1 n h w c').to(device=device)
  pcs = einops.rearrange(torch.from_numpy(pcs), 'n h w c -> 1 1 n h w c').to(device=device)
  rbs = einops.rearrange(torch.from_numpy(rbs), 'c -> 1 1 c').to(device=device)


  actions = torch.zeros((0, act_dim+1), device=device) #original act_dim + 1 ignore_collision
  options = torch.zeros((0, option_dim), device=device)

  episode_return, episode_length = 0, 0
  state = torch.clone(states)
  pc = torch.clone(pcs)
  rb = torch.clone(rbs)

  for t in range(max_ep_len):
    # add dummy action
    actions = torch.cat([actions, torch.zeros((1, act_dim+1), device=device)], dim=0) #original act_dim + 1 ignore_collision
    options = torch.cat([options, torch.zeros((1, option_dim), device=device)], dim=0)
    timesteps = torch.arange(t,t+1,device=device).reshape(1, -1)
    action = torch.zeros((1, act_dim), device=device)
    # print("Eval:-- lang_goal_emb shape: ",lang_goal_emb.shape)
    # print("Eval:-- lang_token_emb shape: ",lang_token_embs.shape)
    # print("Eval:-- state shape: ",state.shape)
    # print("Eval:-- pc shape: ",pc.shape)
    # print("Eval:-- rb shape: ",rb.shape)

    action, ops_idx = get_option_action(
      model, state, pc, rb, options, timesteps, lang_goal_emb,lang_token_embs, **kwargs)
    options_list.append(ops_idx.cpu().item())
    for w in all_tokens:
      words_dict.setdefault(ops_idx.cpu().item(),[]).append(w)
    # action = get_action(model,states,pcs,lm_input['input_ids'],lm_input['attention_mask'],timesteps)

    actions[-1] = torch.Tensor(action)

    # print("action & actions shape: ",action.shape,"*****"*10, actions.shape)

    # ignore_collisions = torch.Tensor([1.0]).to(states.device)
    # action = torch.cat([action.squeeze(), ignore_collisions]) # RLBench-peract/action_modes
    # action = action.detach().cpu().numpy()


    # if model.decision_transformer.discrete:
    #   actions[-1] = torch.nn.functional.one_hot(action, num_classes=act_dim)
    # else:
    #   pass
    #   #need update
      
    #   # action = torch.clamp(action, torch.from_numpy(env.action_space.low).to(
    #   #   device), torch.from_numpy(env.action_space.high).to(device))
    #   # actions[-1] = action

    # if t >= len(action_target):
    #   break
    # action = action_target[t]
    # while (action==0).all():
    #   action_target=torch.cat((action_target[:t],action_target[t+1:]),dim=0)
    #   action = action_target[t]
    # ignore_col = torch.Tensor([1.0]).to(device)
    # action = torch.cat([action, ignore_col])
    # print("target action: ",action_target[t],"="*20,action)

    try:
      state, reward, done = env.step(action)

      state,pc,rbs = extract_rgb_ob(state,cameras)
      # print("state shape: ", state.shape)
      if render:
        images.append(state[0].squeeze())

      episode_return += reward
      episode_length += 1
    
    except (IKError, ConfigurationPathError, InvalidActionError) as e:
      print(e)
      done = True
      reward = 0.0
      fail_sig = True
    
    if t == max_ep_len-1:
      done = True
      reward = 0.0
      fail_sig = True
      print("exceed the maximum episode length!!!")
        
    # if isinstance(state_dim, tuple):
    #   cur_state = torch.from_numpy(state).to(device=device).reshape(1, *state_dim)
    # else:
    #   cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
    # states = torch.cat([states, cur_state], dim=0).float()
    # timesteps = torch.cat(
    #   [timesteps,torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

    

    if done:
      break
    state = einops.rearrange(torch.from_numpy(state),'n h w c -> 1 1 n h w c').to(device=device)
    # print("states & state shape: ",states.shape,"@"*20,state.shape)
    # states = torch.cat([states,state],dim=1)
    pc = einops.rearrange(torch.from_numpy(pc), 'n h w c -> 1 1 n h w c').to(device=device)
    # pcs = torch.cat([pcs, pc], dim=1)
    rb = einops.rearrange(torch.from_numpy(rbs), 'c -> 1 1 c').to(device=device)

  # images = np.stack(images,axis=0)
  # images = torch.from_numpy(images)*255
  # images = einops.rearrange(images.to(torch.float32),'t n h w c -> (t n) h w c')
  # print("image type: ",type(images))
  # print("images shape: ",images.shape,"       ",torch.max(images))
  # print("image mean",torch.mean(images))
  return episode_return, episode_length, options_list, images, words_dict,fail_sig



def get_option_action(model, states, point_cloud, robot_state, options, timesteps,lang_goal_emb, lang_token_embs, **kwargs):
  action,op_idx = model.get_action(
    lang_goal_emb.to(dtype=torch.float32),
    lang_token_embs.to(dtype=torch.float32),
    states,
    point_cloud,
    robot_state,
    timesteps,
    options=options
  )
  return action, op_idx

def get_action(model,states,point_cloud,word_ids,word_mask,timesteps):
  attention_mask = torch.ones(1, states.shape[1]).to(device=states.device).reshape(1, -1)
  action = model(word_ids, word_mask, states, point_cloud, kf=None, gd_actions=None, timesteps=timesteps, attention_mask=attention_mask)
  action = action.squeeze(0)[-1]

  return action