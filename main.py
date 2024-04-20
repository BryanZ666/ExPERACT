import hydra
from omegaconf import DictConfig, OmegaConf

import argparse

import time
import torch
from torch.utils.data import WeightedRandomSampler

import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import re
import sys
import shutil
import pickle
import ast
import wandb
import datetime

from const import *

from rlbench.demo import Demo
from typing import List
from utils import *
from utils import _keypoint_discovery
from PIL import Image
import tqdm

from rlbench.utils import get_stored_demos
from rlbench.backend.observation import Observation
from torch.utils.data import DataLoader, Dataset

from transformers import DistilBertTokenizer, DistilBertModel

import gym
import hydra
from omegaconf import DictConfig, OmegaConf


from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity,EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget,OpenDrawer,PushButton

from hex import Hex
from basemodel import *
from hive.transformer_unet import TransformerUNet
from trainer import Trainer
# from dataloader_disk_store import load_traj,load_multiple_task_demo,ExpDataset
from dataloader import load_traj,load_multiple_task_demo,ExpDataset
from eval import eval_multiple_task_demo

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg : DictConfig):
  cfg.hydra_base_dir = os.getcwd()
  print(OmegaConf.to_yaml(cfg))
  args = cfg

  args.exp_name = f'{args.project_name}-{args.train_dataset.num_trajectories}-{args.method}'
  args.savepath = f'{args.hydra_base_dir}/{args.savedir}/{args.exp_name}-{datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}'

  #online visualization tool W&B
  if args.wandb:
    wandb.init(
      name=args.exp_name,
      group=args.method,
      project=f'hrl_{args.env.name}',
      config=dict(args),
      entity='il-exp'
    )

  if not os.path.isdir(args.savepath):
    os.makedirs(args.savepath, exist_ok=True)

  obs_config = create_obs_config(args.env.cameras,[IMAGE_SIZE,IMAGE_SIZE])

  batch_size = args.batch_size

  data_path = os.path.join(DATA_FOLDER, EPISODES_FOLDER)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(), gripper_action_mode=Discrete()),
        obs_config=obs_config,dataset_root=data_path,headless=True)
  env.launch()

  # using dataloader.py
  num_replay,max_len,max_num_kf,ori_weight,task_weight = load_multiple_task_demo(data_path=data_path, num_demos=args.num_demo, task_list=args.env.task, 
    cameras=args.env.cameras, img_size=IMAGE_SIZE,episode_length=25,replay_path=args.replay_path)
  ds = ExpDataset(num_replay=num_replay,max_len=max_len,max_num_kf=max_num_kf,replay_path=args.replay_path,device=device,ori_weight=ori_weight,task_weight=task_weight)

  sampler = WeightedRandomSampler(ds.weight, len(ds.weight), replacement=True)

  # dl = DataLoader(dataset=ds, batch_size=batch_size, sampler=sampler,shuffle=True, pin_memory=True, drop_last=True)
  dl = DataLoader(dataset=ds, batch_size=batch_size, sampler=sampler)


  if not args.env.eval_offline:
    val_loader = None
  else:
    val_dataset = ExpDataset(all_trajs,subsample_fre=1)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,shuffle=True, pin_memory=True, drop_last=True)

  model = Hex(device=device,args=args,max_length=512)

  start_iter = 1
  if args.resume:
    args.warmup_steps = 0
    #checkpoint = trainer.load(args.checkpoint_path)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    start_iter = checkpoint['iter_num'] + 1
    assert ds.max_length == checkpoint[
      'train_dataset_max_length'], f"Expected max length of dataset to be {ds.max_length} but got {checkpoint['train_dataset_max_length']}"
    
  if args.load_options:
    checkpoint = torch.load(args.checkpoint_path)
    checkpoint = checkpoint['model']
    state_dict = {k:v for k,v in checkpoint.items() if k.startswith('option_selector.Z')}
    loaded = model.load_state_dict(state_dict, strict=False)
    assert loaded.unexpected_keys == []   ## simple check
    if args.freeze_loaded_options:
      for name, param in model.named_parameters():
        if name.startswith('option_selector.Z'):
          param.requires_grad = False
      assert not model.option_selector.Z.project_out.bias.requires_grad   ## simple check

  if args.parallel:
    model = torch.nn.DataParallel(model).to(device)
  else:
    model = model.to(device)

  trainer_args = dict(args.trainer)

  trainer = Trainer(
      model=model,
      train_loader=dl,
      env=env,
      val_loader=val_loader,
      eval_episode_factor=2,
      skip_words=args.env.skip_words,
      device=device,
      render_path=args.render_path,
      cameras = args.env.cameras,
      **trainer_args
  )


  steps = 0
  for iter_num in range(start_iter, start_iter + args.max_iters):
    if steps > args.total_steps:
      break
    outputs,steps = trainer.train_iteration(steps=steps,
      iter_num=iter_num, print_logs=False, eval_render=args.render,total_steps=args.total_steps,log_interval=args.log_interval)

    print("--"*20)
    # print("t1: %f, t2: %f, t3: %f" %(ds.t1,ds.t2,ds.t3))
    

    if iter_num % args.save_interval == 0:
      filepath = f'{args.savepath}/model_{iter_num}.ckpt'
      trainer.save(iter_num, filepath, args)

  env.shutdown()
  ds.shutdown()



@hydra.main(version_base=None, config_path="conf", config_name="config")
def evaluation(cfg : DictConfig):
  cfg.hydra_base_dir = os.getcwd()
  print(OmegaConf.to_yaml(cfg))
  args = cfg

  args.exp_name = f'{args.project_name}-{args.train_dataset.num_trajectories}-{args.method}'
  args.savepath = f'{args.hydra_base_dir}/{args.savedir}/{args.exp_name}'



  if not os.path.isdir(args.savepath):
    os.makedirs(args.savepath, exist_ok=True)

  obs_config = create_obs_config(args.env.cameras,[IMAGE_SIZE,IMAGE_SIZE])
  batch_size = args.batch_size

  # data_path = os.path.join(DATA_FOLDER, EPISODES_FOLDER)
  data_path = os.path.join(DATA_FOLDER, EVAL_FOLDER)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(), gripper_action_mode=Discrete()),
        obs_config=obs_config,dataset_root=data_path,headless=True)
  env.launch()

  # using dataloader.py
  num_replay,max_len,max_num_kf,ori_weight,task_weight = load_multiple_task_demo(data_path=data_path, num_demos=1, task_list=args.env.task, 
    cameras=args.env.cameras, img_size=IMAGE_SIZE,episode_length=25,replay_path=args.replay_path)
  eval_ds = ExpDataset(num_replay=num_replay,max_len=max_len,max_num_kf=max_num_kf,replay_path=args.replay_path,device=device,ori_weight=ori_weight,task_weight=task_weight)


  sampler = WeightedRandomSampler(eval_ds.weight, len(eval_ds.weight), replacement=True)

  # dl = DataLoader(dataset=ds, batch_size=batch_size, sampler=sampler,shuffle=True, pin_memory=True, drop_last=True)
  eval_dl = DataLoader(dataset=eval_ds, batch_size=batch_size, sampler=sampler)

  num_layers = 4
  latent_img_side_size = int(IMAGE_SIZE/(2**num_layers))

  trainer_args = dict(args.trainer)
  trainer_args['num_eval_episodes'] = 25

  folder_path = args.savepath

  # Get a list of all files in the folder
  model_files = [file for file in os.listdir(folder_path) if file.endswith('.ckpt')]

  # Sort the model files to ensure loading in the correct order
  model_files.sort()
  eval_model = None
  for model_file in model_files:
    match = re.match(r"model_(\d+)\.ckpt",model_file)
    iter_num = int(match.group(1))
    eval_model = Hex(device=device,args=args,max_length=512)
    if args.parallel:
      eval_model = torch.nn.DataParallel(eval_model).to(device)
    else:
      eval_model = eval_model.to(device)    
    eval_trainer = Trainer(
        model=eval_model,
        train_loader=eval_dl,
        env=env,
        val_loader=eval_dl,
        eval_episode_factor=2,
        skip_words=args.env.skip_words,
        device=device,
        render_path=args.render_path,
        cameras = args.env.cameras,
        **trainer_args
    )
    model_path = os.path.join(folder_path, model_file)
    eval_trainer.load(model_path)
    eval_trainer.model.eval()


    action_target = torch.zeros((1,args.env.action_dim)).detach()

    for ep in range(args.eval_ep):
      eva_outputs = eval_trainer.evaluate(
        "eval", render=True,render_path=args.render_path,action_target=action_target)
      eval_filename = f"{iter_num}_{ep}_tasks_success_rate.csv"
      eval_csv_path = os.path.join(folder_path, eval_filename)
      with open(eval_csv_path, mode='w', newline='') as file:
          writer = csv.writer(file)
          writer.writerow(eva_outputs["task_list"])
          writer.writerow(eva_outputs["success_rate"])

    # eval training demos
    train_data_path = os.path.join(DATA_FOLDER, EPISODES_FOLDER)
    eval_multiple_task_demo(data_path=train_data_path, num_demos=args.num_demo, task_list=args.env.task, 
      cameras=args.env.cameras, img_size=IMAGE_SIZE,episode_length=25,eval_model=eval_trainer.model,save_note=iter_num)

  del eval_trainer, eval_model
  env.shutdown()
  eval_ds.shutdown()




if __name__=="__main__":
  evalu = False
  os.environ["DISPLAY"] = ":0"
  os.environ["PYOPENGL_PLATFORM"] = "egl" 

  if not evalu:
    train()
  else:
    evaluation()
