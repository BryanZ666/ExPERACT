import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import imageio
import numpy as np
from tqdm import tqdm
import os
import imageio
import wandb
from rlbench.tasks import *



# from env import BaseWrapper, LorlWrapper, BabyAIWrapper
from eval import eval_episode,viz_matrix
from utils import *
# from viz import get_tokens, viz_matrix, plot_hist

EVA_TASK = [OpenDrawer,MeatOffGrill,SlideBlockToColorTarget,\
  CloseJar,TurnTap,StackBlocks,LightBulbIn,PushButtons]
# EVA_TASK = [OpenDrawer,MeatOffGrill]

class Trainer:

  def __init__(self, model, train_loader, env=None, val_loader=None,
      state_il=True, eval_episode_factor=2, eval_every=5, num_eval_episodes=10,
      skip_words=None, device='cuda',render_path = None,cameras = None):
    self.model = model
    # self.optimizer = optimizer
    self.train_loader = train_loader
    self.env = env
    self.val_loader = val_loader
    self.state_il = state_il  # do Imitation learning on states too?
    # self.scheduler = scheduler
    self.eval_episode_factor = eval_episode_factor
    self.eval_every = eval_every
    self.num_eval_episodes = num_eval_episodes
    self.device = device
    self.skip_words = skip_words
    self.render_path = render_path
    self.cameras = cameras
    self.start_time = time.time()

  def train_iteration(self, steps,iter_num, print_logs=False, eval_render=False,total_steps=100000,log_interval=1):
    train_losses, action_losses, action_errors, state_losses, option_losses = [], [], [], [], []
    state_rc_losses, lang_rc_losses = [], []
    commitment_losses = []
    entropies, lang_entropies, mutual_info = [], [], []
    logs = dict()
    if steps > total_steps:
      return logs,steps

    train_start = time.time()

    model = self.model
    if hasattr(self.model, 'module'):
      model = self.model.module

    for lang_choice, lang_goal_emb, lang_token_embs, states, pcs, rbs, actions, trans_idx,\
        rot_idx, dones, timesteps, attention_mask in self.train_loader:
      # print(lang_choice)  #tuple (1,)

      act_dim = actions.shape[-1]
      model.train()
      if steps > total_steps:
        break
      
      states = states.float().to(self.device) #B,T,N,H,W,C
      pcs = pcs.float().to(self.device)
      rbs = rbs.float().to(self.device)
      actions = actions.float().to(self.device) #B,T,act_dim
      trans_idx = trans_idx.to(dtype=torch.int32,device=self.device)
      rot_idx = rot_idx.to(dtype=torch.int32,device=self.device)
      timesteps = timesteps.to(dtype=torch.int32,device=self.device)
      dones = dones.to(dtype=torch.int32,device=self.device)
      attention_mask = attention_mask.to(dtype=torch.int32,device=self.device)

      action_target = torch.clone(actions).detach()

      assert states.shape[0] ==pcs.shape[0] == actions.shape[0] == \
        timesteps.shape[0] ==dones.shape[0] ==attention_mask.shape[0]
      assert states.shape[1] == pcs.shape[1] == actions.shape[1] == \
        timesteps.shape[1] ==dones.shape[1] ==attention_mask.shape[1] 

      # action_preds,commitment_loss = self.model(steps,lang_goal_emb, lang_token_embs, \
      #   states,pcs, rbs, actions, trans_idx, rot_idx, timesteps, attention_mask=attention_mask)
      loss = self.model(steps,lang_goal_emb, lang_token_embs, \
        states,pcs, rbs, actions, trans_idx, rot_idx, timesteps, attention_mask=attention_mask)
      # act_dim = action_preds.shape[-1]
      # num_cam = states.shape[2]

      #action_target = action_target.repeat([1,num_cam,1])
      #attention_mask = attention_mask.repeat([1,num_cam])
      # action_preds = action_preds.reshape(-1, act_dim)
      # action_target = action_target.reshape(-1, act_dim)

      # with torch.no_grad():
      #   dev_act = torch.sub(action_preds,action_target,alpha=1)
      #   zero_act = torch.zeros_like(dev_act)
      #   mseloss = nn.MSELoss()
      #   printing_out = mseloss(dev_act,zero_act)
      #   print(printing_out)
      #   if torch.isnan(printing_out).all():
      #     print(action_preds)
      #     print("=*"*10)
      #     print(action_target)
      #     print("state shape:     ", states.shape)

      # loss = imitation_loss(action_preds=action_preds,action_targets=action_target)+commitment_loss
      # print(loss)

      # self.optimizer.zero_grad()
      # loss.backward()
      # self.optimizer.step()
      # if self.scheduler is not None:
      #   self.scheduler.step(steps)

      with torch.no_grad():
        train_losses.append(loss.detach().cpu().item())
        action_losses.append(loss.detach().cpu().item())
        # state_losses.append(state_loss.detach().cpu().item())
        # option_losses.append(options_loss.detach().cpu().item())
        # state_rc_losses.append(state_rc_loss.detach().cpu().item())
        # lang_rc_losses.append(lang_rc_loss.detach().cpu().item())
        # commitment_losses.append(commitment_loss.detach().cpu().item())
        # entropies.append(entropy.detach().cpu().item())
        # lang_entropies.append(lang_entropy.detach().cpu().item())
        # mutual_info.append(entropies[-1] - lang_entropies[-1])
      # if steps>0 and steps % self.eval_every == 0:
      #   eval_start = time.time()

      #   self.model.eval()
      #   eval_outputs = self.evaluate(steps, render=eval_render,render_path=self.render_path,action_target=action_target)
      #   for k, v in eval_outputs.items():
      #     logs[f'evaluation/{k}'] = v
      #   logs['time/evaluation'] = time.time() - eval_start
      

      steps +=1      

    logs['time/training'] = time.time() - train_start


    logs['time/total'] = time.time() - self.start_time
    logs['training/train_loss_mean'] = np.mean(train_losses)
    logs['training/train_loss_std'] = np.std(train_losses)

    logs['training/action_pred_loss'] = np.mean(action_losses)
    # logs['training/state_pred_loss'] = np.mean(state_losses)
    # logs['training/options_pred_loss'] = np.mean(option_losses)
    # logs['training/state_rc_loss'] = np.mean(state_rc_losses)
    # logs['training/lang_rc_loss'] = np.mean(lang_rc_losses)
    # logs['training/commitment_loss'] = np.mean(commitment_losses)
    # logs['training/entropy'] = np.mean(entropies)
    # logs['training/lang_entropy'] = np.mean(lang_entropies)
    # logs['training/MI'] = np.mean(entropies) - np.mean(lang_entropies)
    # logs['training/lr'] = self.optimizer.param_groups[0]['lr']  # DT lr
    # logs['training/lm_lr'] = self.optimizer.param_groups[1]['lr']  # Language model lr
    # logs['training/os_lr'] = self.optimizer.param_groups[2]['lr']  # option selector lr

    if print_logs:
      print('=' * 80)
      print(f'Iteration {iter_num}')
      for k, v in logs.items():
        print(f'{k}: {v}')
    
    if steps>0 and steps % log_interval == 0:
      wandb.log(logs, step=steps)

    return logs,steps

  def evaluate(self, iter_num, render=False, max_ep_len=100, render_path='./viz/', render_freq=1,action_target=None):
    eval_tasks = EVA_TASK
    device = self.device
    model = self.model.to(device)
    if hasattr(self.model, 'module'):
      model = self.model.module

    words_dict = {}

    # setting this to be sufficiently large
    #max_ep_len = self.eval_episode_factor * self.train_loader.dataset.max_length
    # max_ep_len = self.train_loader.dataset.max_length*2

    if render:
      if not os.path.isdir(render_path):
        os.makedirs(render_path, exist_ok=True)

    task_list, succ_rate_list = [], []

    

    for t in eval_tasks:
      # self.env.launch()
      task = self.env.get_task(t)
      task_name = task.get_name()
      count = self.num_eval_episodes
      for i in tqdm(range(1, self.num_eval_episodes+1)):
        # task.set_variation(np.random.randint(0,task.variation_count()))
        
        episode_return, episode_length, options_list, images, words_dict,fail_sig = eval_episode(
          task, model, max_ep_len, words_dict, render, device, render_path=render_path,
          render_freq=render_freq, iter_num=iter_num, seed=i%20,action_target=action_target,cameras=self.cameras)

        if fail_sig:
          count -=1

        if render:
          r = f'{task_name}_{iter_num}_{i}'
          imageio.mimsave(f'{render_path}/episode_{r}.gif', images,fps=8)

          with open(f'{render_path}/episode_{r}_options.txt', 'w') as fp:
            fp.write(str(options_list))
        del images
      print(task_name," success rate: ",count," / ",self.num_eval_episodes)
      task_list.append(task_name)
      succ_rate_list.append((count/self.num_eval_episodes))
    torch.cuda.empty_cache()
    viz_matrix(words_dict, 16, f"{render_path}/{iter_num}_")


    #instr_table = wandb.Table(data=instr_wise_stats, columns=["Instruction", "Success Rate"])
    #rephrasal_table = wandb.Table(data=rephrasal_wise_stats, columns=["Rephrasal type", "Success Rate"])

    metrics = {
      f'task_list': task_list,
      f'success_rate': succ_rate_list}
    
    return metrics

  def save(self, iter_num, filepath, config):
    model = self.model
    if hasattr(self.model, 'module'):
        model = self.model.module

    torch.save({'opt_model': model.option_selector.state_dict(),
                'act_model':model.act_transformer._q.state_dict(),
                'model':model.state_dict(),
                'optimizer': model.optimizer.state_dict(),
                'scheduler': model.scheduler.state_dict(),
                'iter_num': iter_num,
                'train_dataset_max_length': self.train_loader.dataset.max_length,
                'config': config}, filepath)

  def load(self, filepath):
    checkpoint = torch.load(filepath)
    if hasattr(self.model, 'module'):
      self.model.module.load_state_dict(checkpoint['model'])
      self.model.module.option_selector.load_state_dict(checkpoint['opt_model'])
      self.model.module.act_transformer._q.load_state_dict(checkpoint['act_model'])
    else:
      self.model.load_state_dict(checkpoint['model'])
      self.model.option_selector.load_state_dict(checkpoint['opt_model'])
      self.model.act_transformer._q.load_state_dict(checkpoint['act_model'])
    if hasattr(self.model, 'optimizer'):
      self.model.optimizer.load_state_dict(checkpoint['optimizer'])
    if hasattr(self.model, 'scheduler'):
      self.model.scheduler.load_state_dict(checkpoint['scheduler'])
    return {'iter_num': checkpoint['iter_num'], 'train_dataset_max_length': checkpoint['train_dataset_max_length'], 'config': checkpoint['config']}



