from typing import Any, Dict, IO, List, Tuple

import numpy as np
import pandas as pd
import pickle
import torch
from torchvision import transforms
import torchvision.transforms.functional as transforms_f
from torch.utils.data import Dataset, DataLoader
import os
from os.path import join

from utils import *
from utils import _keypoint_discovery

from tqdm import tqdm
from rlbench.utils import get_stored_demos

from const import *
from clip.core.clip import build_model, load_clip, tokenize


def _get_action(obs_tp1: Observation,
                rlbench_scene_bounds: List[float] = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6],
                voxel_sizes: List[int] = [100],
                bounds_offset: List[float]=[0.15],
                rotation_resolution: int=5,
                crop_augmentation: bool=True):
    quat = normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = quaternion_to_discrete_euler(quat, rotation_resolution)
    rot_and_grip_indicies = disc_rot.tolist()
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])

    attention_coordinate = obs_tp1.gripper_pose[:3]
    trans_indicies = []
    bounds = np.array(rlbench_scene_bounds)
    for depth, vox_size in enumerate(voxel_sizes): # only single voxelization-level is used in PerAct
        if depth > 0:
            if crop_augmentation:
                shift = bounds_offset[depth - 1] * 0.75
                attention_coordinate += np.random.uniform(-shift, shift, size=(3,))
            bounds = np.concatenate([attention_coordinate - bounds_offset[depth - 1],
                                     attention_coordinate + bounds_offset[depth - 1]])
        index = point_to_voxel_index(
            obs_tp1.gripper_pose[:3], vox_size, bounds)
        trans_indicies.extend(index.tolist())
        res = (bounds[3:] - bounds[:3]) / vox_size
        attention_coordinate = bounds[:3] + res * index

    grip = float(obs_tp1.gripper_open)
    return trans_indicies, rot_and_grip_indicies, np.concatenate(
        [obs_tp1.gripper_pose, np.array([grip])])

def load_traj(data_path, num_demos, task, cameras, img_size, episode_length, save_index, replay_path='./rp/',
    sample_fre =10,hist_size = 4):
  obs_cfg = create_obs_config(cameras,[img_size,img_size])
  max_len = 0
  max_num_kf = 0
  num_stored = 0
  num_obs_list = []
  for idx in range(num_demos):
    
    demo = get_stored_demos(amount=1, image_paths=False,
      dataset_root=data_path,
      variation_number=-1, task_name=task,
      obs_config=obs_cfg,
      random_selection=False,
      from_episode_number=idx)[0]

    episode_keypoints = _keypoint_discovery(demo)
    # episode_keypoints.insert(0,0)
    
    last_kp = episode_keypoints[-1]

    demo_len = len(demo)
    traj_ob = []
    traj_pcs = []
    traj_act = []
    traj_rbstate = []
    traj_trans_idx = []
    traj_rot_idx = []
    #descs consists of multiple text expression of the task.
    descs = demo._observations[0].misc['descriptions']
    # descs = descs[:1]

    

    for ts in range(demo_len):
      if ts % sample_fre !=0:
        continue

      ob_sq = []
      pcs_sq = []
      act_sq = []
      rbstate_sq = []
      trans_idx_sq = []
      rot_idx_sq = []
      trajs = {"states": [], "point_cloud": [], "robot_state": [], "actions": [], "trans_idx": [], "rot_idx":[],
        "timesteps": [], "attention_mask": [], "language": [], "dones": [], "keyframes": []}

      pre_keyframe = 0
      
      epkp = episode_keypoints[:]
      
      while len(epkp)>0 and ts>=epkp[0]:
        pre_keyframe = epkp[0]
        epkp=epkp[1:]
      if len(epkp)==0:
        break
      if ts<pre_keyframe+hist_size: # avoid the beginning of each keyframe repeat many times
        ts = min(epkp[0]-1,pre_keyframe+hist_size+1)

      # load previous obs for option inference
      history_list = sorted(random.sample(range(pre_keyframe,ts),hist_size)) if epkp[0]-1 - pre_keyframe > hist_size else []
      for hist_point in history_list:
        obs_dict = extract_obs(demo[hist_point],0,episode_length=episode_length)
        rbstate = obs_dict["robot_state"]

        ob = []
        pcs = []
        for camera in cameras:
          rgb_name = "%s_%s" % (camera, 'rgb')
          rgb = np.array(obs_dict[rgb_name])

          rgb = np.transpose(rgb, (1, 2, 0))
          ob.append(rgb)

          pc_name = "%s_%s" % (camera, 'point_cloud')
          pc = np.array(obs_dict[pc_name])
          pc = np.transpose(pc, (1, 2, 0))
          pcs.append(pc)

        ob =np.stack(ob)  # shape = (4,128,128,3)
        pcs = np.stack(pcs)

        trans_indicies, rot_and_grip_indicies, action = _get_action(demo[epkp[0]])  # shape = (8,)

        ob_sq.append(ob)
        pcs_sq.append(pcs)
        rbstate_sq.append(rbstate)
        act_sq.append(action)
        trans_idx_sq.append(trans_indicies)
        rot_idx_sq.append(rot_and_grip_indicies)

      
      # load current obs and following keyframes
      init_obs = demo[ts]
      epkp.append(epkp[-1])
      for keystage, keypoint in enumerate(epkp):

        obs_dict = extract_obs(init_obs,keystage,episode_length=episode_length)
        # obs_dict = extract_obs(demo._observations[ts],ts,episode_length=episode_length)

        rbstate = obs_dict["robot_state"]

        ob = []
        pcs = []
        for camera in cameras:
          rgb_name = "%s_%s" % (camera, 'rgb')
          rgb = np.array(obs_dict[rgb_name])
          # rgb = transforms_f.normalize(
          #   torch.from_numpy(rgb).to(torch.float32), 
          #   [0.485, 0.456, 0.406], 
          #   [0.229, 0.224, 0.225]
          # )

          rgb = np.transpose(rgb, (1, 2, 0))
          ob.append(rgb)

          pc_name = "%s_%s" % (camera, 'point_cloud')
          pc = np.array(obs_dict[pc_name])
          pc = np.transpose(pc, (1, 2, 0))
          pcs.append(pc)

        ob =np.stack(ob)  # shape = (4,128,128,3)
        pcs = np.stack(pcs)
        # trans_indicies, rot_and_grip_indicies, action = _get_action(demo[ts+1])  # shape = (8,)

        kp_idx = keypoint if len(epkp)>0 else last_kp
        # print("At timestep %d, extract action from timestep %d " %(ts,kp_idx))
        trans_indicies, rot_and_grip_indicies, action = _get_action(demo[kp_idx])  # shape = (8,)

        #print(ob.shape,action.shape)

        ob_sq.append(ob)
        pcs_sq.append(pcs)
        rbstate_sq.append(rbstate)
        act_sq.append(action)
        trans_idx_sq.append(trans_indicies)
        rot_idx_sq.append(rot_and_grip_indicies)

        init_obs = demo[keypoint]

      trajs["states"]=np.array(ob_sq,dtype=np.float32)
      trajs["point_cloud"]=np.array(pcs_sq,dtype=np.float32)
      trajs["robot_state"]=np.array(rbstate_sq,dtype=np.float32)
      trajs["actions"]=np.array(act_sq,dtype=np.float32)
      trajs["trans_idx"]=np.array(trans_idx_sq,dtype=np.long)
      trajs["rot_idx"]=np.array(rot_idx_sq,dtype=np.long)
      trajs["language"]=np.array(descs,dtype=object)
      trajs["dones"]= np.array([0]*(len(epkp)+len(history_list)-1)+[1],dtype=np.long)
      trajs["keyframes"]=np.array(epkp[:-1])
      trajs["timesteps"] = np.arange(0,len(epkp)+len(history_list))
      trajs["attention_mask"] = np.ones(len(epkp)+len(history_list))
      s_idx = save_index+num_stored
      with open(join(replay_path, '%d.replay' % s_idx), 'wb') as f:
        pickle.dump(trajs, f)
      num_stored +=1
      num_obs_list.append(len(epkp)+len(history_list))

      max_len = max(max_len,len(epkp)+len(history_list))
      max_num_kf = max(max_num_kf,len(episode_keypoints))
    
  total_obs_number = sum(num_obs_list)
  total_obs_list  = [total_obs_number]*len(num_obs_list)  

  return save_index+num_stored,max_len,max_num_kf,num_obs_list,total_obs_list


def load_multiple_task_demo(data_path, num_demos, task_list, cameras, img_size,episode_length,replay_path='./rp/'):
  idx = 0 
  max_len = 0
  max_num_kf = 0
  task_weight =[]
  ori_weight = []
  for task in task_list:
    idx,traj_len,num_kf,ori_w,wei = load_traj(data_path, num_demos, task, cameras, img_size,episode_length,idx,replay_path)
    max_len = max(max_len,traj_len)
    max_num_kf = max(max_num_kf,num_kf)
    task_weight.extend(wei)
    ori_weight.extend(ori_w)

  return idx,max_len,max_num_kf,ori_weight,task_weight


    
class ExpDataset(Dataset):
  def __init__(self, num_replay, max_len, max_num_kf, replay_path = "./rp/", device=None,ori_weight=[],task_weight=[]):
    self.num_replay = num_replay
    self.replay_path = replay_path
    self.max_length = max_len
    self.max_num_kf = max_num_kf
    self.device = device

    model, _ = load_clip('RN50', jit=False, device=device)
    clip_model = build_model(model.state_dict())
    clip_model.to(device=device)
    self.clip_model = clip_model
    del model

    self.test_idx = 0

    # self.weight = [1.0/tw for tw in task_weight]
    self.weight = [1.0/(task_weight[i]+ori_weight[i]) for i in range(len(task_weight))]
    self.original_weight = ori_weight
    
    self.t1 = 0
    self.t2 = 0
    self.t3 = 0
    # assert False
  
  def __len__(self):
    return self.num_replay

  def __getitem__(self, index):
    # index = self.test_idx
    # self.test_idx = (self.test_idx +1)%self.num_replay
    with open(join(self.replay_path, '%d.replay' % index), 'rb') as f:
        trajectories = pickle.load(f)
    traj_states = np.array(trajectories["states"]) # T, N, H, W, C
    traj_point_cloud = np.array(trajectories["point_cloud"])
    traj_rbstate = np.array(trajectories["robot_state"])
    traj_act = np.array(trajectories["actions"])
    traj_trans_idx = np.array(trajectories["trans_idx"])
    traj_rot_idx = np.array(trajectories["rot_idx"])
    
    traj_dones = np.array(trajectories["dones"])


    # lang_choice = str(trajectories["language"][0]) 
    lang_choice = str(np.random.choice(trajectories["language"]))
    print(lang_choice)

    if "grill" in lang_choice:
      self.t1 +=  self.original_weight[index]
    elif "item" in lang_choice:
      self.t3 +=self.original_weight[index]
    else:
      self.t2 +=self.original_weight[index]

    tokens = tokenize([lang_choice]).numpy()
    token_tensor = torch.from_numpy(tokens).to(device=self.device)
    sentence_emb, token_embs = self.clip_model.encode_text_with_embeddings(token_tensor)
    lang_goal_emb = sentence_emb[0].float().detach().cpu().numpy()
    lang_token_embs = token_embs[0].float().detach().cpu().numpy()


    timesteps = np.array(trajectories["timesteps"])
    attention_mask = np.array(trajectories["attention_mask"])

    # ps = pad(traj_states,self.max_num_kf+1,axis=0)
    # ppc = pad(traj_point_cloud, self.max_num_kf+1, axis=0)
    # p_rb = pad(traj_rbstate,self.max_num_kf+1,axis=0)
    # pa = pad(traj_act,self.max_num_kf+1,axis=0)
    # pdones = pad(traj_dones,self.max_num_kf+1,axis=0)
    # ptimes = pad(timesteps,self.max_num_kf+1,axis=0)
    # patt = pad(attention_mask,self.max_num_kf+1,axis=0)

    while (traj_states[-1]==0).all():
      traj_states=traj_states[:-1]
      traj_point_cloud = traj_point_cloud[:-1]
      traj_rbstate = traj_rbstate[:-1]
      traj_act = traj_act[:-1]
      traj_trans_idx = traj_trans_idx[:-1]
      traj_rot_idx = traj_rot_idx[:-1]
      traj_dones = traj_dones[:-1]
      timesteps = timesteps[:-1]
      attention_mask = attention_mask[:-1]

    ps = traj_states
    ppc = traj_point_cloud
    p_rb = traj_rbstate
    pa = traj_act
    ptransidx = traj_trans_idx
    protidx = traj_rot_idx
    pdones = traj_dones
    ptimes = timesteps
    patt = attention_mask


    return (
      lang_choice,
      lang_goal_emb,
      lang_token_embs,
      ps,
      ppc,
      p_rb,
      pa,
      ptransidx,
      protidx,
      pdones,
      ptimes,
      patt
    )

  def shutdown(self):
    # Safely delete replay
    for f in [f for f in os.listdir(self.replay_path) if '.replay' in f]:
      os.remove(join(self.replay_path, f))
  

