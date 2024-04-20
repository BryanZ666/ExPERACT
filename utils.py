import numpy as np
import torch
from rlbench.backend.const import DEPTH_SCALE
from rlbench.demo import Demo
from scipy.spatial.transform import Rotation
from rlbench.backend.observation import Observation
from rlbench import CameraConfig, ObservationConfig
from pyrep.const import RenderMode
from typing import List
import random

import torch.nn.functional as F
import itertools

from einops import rearrange, repeat

import trimesh
import pyrender
from pyrender.trackball import Trackball

REMOVE_KEYS = ['joint_velocities', 'joint_positions', 'joint_forces',
               'gripper_open', 'gripper_pose',
               'gripper_joint_positions', 'gripper_touch_forces',
               'task_low_dim_state', 'misc']

SCALE_FACTOR = DEPTH_SCALE
DEFAULT_SCENE_SCALE = 2.0


def imitation_loss(action_preds, action_targets):
    act_loss = F.mse_loss(action_preds, action_targets)
    return act_loss


def reshape_tensor(kfs,input_tensor,interval):
    # get the maximum length of the interval between keyframes
    ba,num_kfs = kfs.shape
    
    reshaped_tensor = []
    index = []

    # maybe need to use the similar strats to ensure the gradient pass through
    for b in range(ba):
        previous_frame = 0
        cnt = 0
        for i in kfs[b]:
            while previous_frame < i:

                tensor_slice = input_tensor[b,previous_frame:min(i,previous_frame+interval)]

                tensor_slice = pad(tensor_slice, interval,axis=0)

                reshaped_tensor.append(tensor_slice)

                cnt+=1

                previous_frame = min(i,previous_frame+interval)
        index.append(cnt)
        # mode_option,mode_indice=options[b,previous_keyframe:,:].mode(0)
        # options[b, previous_keyframe:, :] = mode_option.clone().detach()
    reshaped_tensor = torch.stack(reshaped_tensor)

    return reshaped_tensor,index

def add_keyframes_info(kfs,states,point_cloud,robot_state,actions,trans_idx,rot_idx,timesteps,dones,attention_mask,hist_size=0):
    # get the maximum length of the interval between keyframes
    while (kfs[:,-1]==0).all():
      kfs = kfs[:,:-1]
    ba,num_kfs = kfs.shape
    ba_s,ts,num_cams,_,_,_ = states.shape
    assert ba == ba_s

    
    reshaped_states = []
    reshaped_pc = []
    reshaped_rb = []
    reshaped_act = []
    reshaped_transidx = []
    reshaped_rotidx = []
    reshaped_ts = []
    reshape_d = []
    reshape_att = []
    weight = []
    kf_weight = []

    for b in range(ba):
        cnt = 0
        pre_keyframe = 0

        for i in range(ts):
            previous_frame = i
            ba_kfs = kfs[b]
            while len(ba_kfs)>0 and i >= ba_kfs[0]:
                pre_keyframe = ba_kfs[0]
                ba_kfs = ba_kfs[1:]
            if len(ba_kfs)==0:
              break
            if i<pre_keyframe+hist_size: # avoid the beginning of each keyframe repeat many times
                # continue
                i = min(ts-1,pre_keyframe+hist_size+1)
                previous_frame = min(ts-1,pre_keyframe+hist_size+1)
            if (states[b,previous_frame]==0).all():
                continue
            cnt +=1

            state_kf = []
            pc_kf = []
            rb_kf = []
            act_kf = []
            trans_kf = []
            rot_kf = []
            ts_kf = []
            d_kf = []
            att_kf = []            


            history_list = sorted(random.sample(range(pre_keyframe,previous_frame),hist_size))
            # history_list = []
            for h_idx in history_list:
                state_kf.append(states[b,h_idx])
                pc_kf.append(point_cloud[b,h_idx])
                rb_kf.append(robot_state[b,h_idx])
                act_kf.append(actions[b,h_idx])
                trans_kf.append(trans_idx[b,h_idx])
                rot_kf.append(rot_idx[b,h_idx])
                ts_kf.append(timesteps[b,h_idx])
                d_kf.append(dones[b,h_idx])
                att_kf.append(attention_mask[b,h_idx])

            state_kf.append(states[b,previous_frame])
            pc_kf.append(point_cloud[b,previous_frame])
            rb_kf.append(robot_state[b,previous_frame])
            act_kf.append(actions[b,previous_frame])
            trans_kf.append(trans_idx[b,previous_frame])
            rot_kf.append(rot_idx[b,previous_frame])
            ts_kf.append(timesteps[b,previous_frame])
            d_kf.append(dones[b,previous_frame])
            att_kf.append(attention_mask[b,previous_frame])

            if len(ba_kfs)>1:
              future_points = random.sample(range(ba_kfs[0],ba_kfs[-1]),2)

            else:
              future_points = []
            future_points = []
            
            ba_kfs = sorted(np.concatenate((ba_kfs,future_points)).astype(int))
            # print("Step: ",i," history index: ",history_list," future keyframe index: ",ba_kfs)

            for k, keypoint in enumerate(ba_kfs):
                previous_frame = keypoint
                if (states[b,previous_frame]==0).all():
                    continue
                state_kf.append(states[b,previous_frame])
                pc_kf.append(point_cloud[b,previous_frame])
                rb_kf.append(robot_state[b,previous_frame])
                act_kf.append(actions[b,previous_frame])
                trans_kf.append(trans_idx[b,previous_frame])
                rot_kf.append(rot_idx[b,previous_frame])
                ts_kf.append(timesteps[b,previous_frame])
                d_kf.append(dones[b,previous_frame])
                att_kf.append(attention_mask[b,previous_frame])
            
            state_kf = torch.stack(state_kf)
            pc_kf = torch.stack(pc_kf)
            rb_kf = torch.stack(rb_kf)
            act_kf = torch.stack(act_kf)
            trans_kf = torch.stack(trans_kf)
            rot_kf = torch.stack(rot_kf)
            ts_kf = torch.stack(ts_kf)
            d_kf = torch.stack(d_kf)
            att_kf = torch.stack(att_kf)

            # print("state shape: %d, keyframe interval: %d"%(state_kf.shape[0],(ba_kfs[0]-pre_keyframe)))
            weight.append(state_kf.shape[0])

            kf_weight.append(ba_kfs[0]-pre_keyframe)

            state_kf = pad(state_kf,num_kfs+hist_size+1,axis=0,mode='post')
            pc_kf = pad(pc_kf,num_kfs+hist_size+1,axis=0,mode='post')
            rb_kf = pad(rb_kf,num_kfs+hist_size+1,axis=0,mode='post')
            act_kf = pad(act_kf,num_kfs+hist_size+1,axis=0,mode='post')
            trans_kf = pad(trans_kf,num_kfs+hist_size+1,axis=0,mode='post')
            rot_kf = pad(rot_kf,num_kfs+hist_size+1,axis=0,mode='post')
            ts_kf = pad(ts_kf,num_kfs+hist_size+1,axis=0,mode='post')
            d_kf = pad(d_kf,num_kfs+hist_size+1,axis=0,mode='post')
            att_kf = pad(att_kf,num_kfs+hist_size+1,axis=0,mode='post')

            reshaped_states.append(state_kf)
            reshaped_pc.append(pc_kf)
            reshaped_rb.append(rb_kf)
            reshaped_act.append(act_kf)
            reshaped_transidx.append(trans_kf)
            reshaped_rotidx.append(rot_kf)
            reshaped_ts.append(ts_kf)
            reshape_d.append(d_kf)
            reshape_att.append(att_kf)

    reshaped_states = torch.stack(reshaped_states).numpy()
    reshaped_pc = torch.stack(reshaped_pc).numpy()
    reshaped_rb = torch.stack(reshaped_rb).numpy()
    reshaped_act = torch.stack(reshaped_act).numpy()
    reshaped_transidx = torch.stack(reshaped_transidx).numpy()
    reshaped_rotidx = torch.stack(reshaped_rotidx).numpy()
    reshaped_ts = torch.stack(reshaped_ts).numpy()
    reshape_d = torch.stack(reshape_d).numpy()
    reshape_att = torch.stack(reshape_att).numpy()

    return reshaped_states,reshaped_pc,reshaped_rb,reshaped_act,reshaped_transidx,\
      reshaped_rotidx,reshaped_ts,reshape_d,reshape_att,weight,kf_weight

def add_keyframes(kfs,states,point_cloud,robot_state,actions,timesteps,dones,attention_mask):
    # get the maximum length of the interval between keyframes
    ba,num_kfs = kfs.shape
    ba_s,ts,num_cams,_,_,_ = states.shape
    assert ba == ba_s
    
    reshaped_states = []
    reshaped_pc = []
    reshaped_rb = []
    reshaped_act = []
    reshaped_ts = []
    reshape_d = []
    reshape_att = []
    index = []

    # maybe need to use the similar strats to ensure the gradient pass through
    for b in range(ba):
        cnt = 0
        pre_keyframe = 0
        for i in range(ts):
            previous_frame = i
            if (states[b,previous_frame]==0).all():
                continue
            cnt +=1
            
            reshaped_states.append(states[b,previous_frame])
            reshaped_pc.append(point_cloud[b,previous_frame])
            reshaped_rb.append(robot_state[b,previous_frame])
            reshaped_act.append(actions[b,previous_frame])
            reshaped_ts.append(timesteps[b,previous_frame])
            reshape_d.append(dones[b,previous_frame])
            reshape_att.append(attention_mask[b,previous_frame])

            ba_kfs = kfs[b]
            while len(ba_kfs)>0 and i >= ba_kfs[0]:
                pre_keyframe = ba_kfs[0]
                ba_kfs = ba_kfs[1:] 

            for k, keypoint in enumerate(ba_kfs):
                previous_frame = keypoint
                if (states[b,previous_frame]==0).all():
                    continue

                cnt+=1
                reshaped_states.append(states[b,previous_frame])
                reshaped_pc.append(point_cloud[b,previous_frame])
                reshaped_rb.append(robot_state[b,previous_frame])
                reshaped_act.append(actions[b,previous_frame])
                reshaped_ts.append(timesteps[b,previous_frame])
                reshape_d.append(dones[b,previous_frame])
                reshape_att.append(attention_mask[b,previous_frame])

        index.append(cnt)

    reshaped_states = torch.stack(reshaped_states).unsqueeze(1)
    reshaped_pc = torch.stack(reshaped_pc).unsqueeze(1)
    reshaped_rb = torch.stack(reshaped_rb).unsqueeze(1)
    reshaped_act = torch.stack(reshaped_act).unsqueeze(1)
    reshaped_ts = torch.stack(reshaped_ts).unsqueeze(1)
    reshape_d = torch.stack(reshape_d).unsqueeze(1)
    reshape_att = torch.stack(reshape_att).unsqueeze(1)
    print(reshaped_states.shape)
    assert False
    return reshaped_states,reshaped_pc,reshaped_rb,reshaped_act,reshaped_ts,reshape_d,reshape_att,index




def round_tensor(t,decimal_p):
    rounded_t = torch.round(t*10**decimal_p)/(10**decimal_p)
    rounded_t= t+rounded_t.detach()-t.detach()
    return rounded_t

def get_max_min_interval(kfs):
    ba,kf_len = kfs.shape
    max_len = 0
    min_len = 999
    for i in range(ba):
        pre_kf = 0
        for j in range(kf_len):
            kf = kfs[i,j]
            if kf > pre_kf:
                max_len = max(max_len,kf-pre_kf)
                min_len = min(min_len,kf-pre_kf)
                pre_kf=kf
    return max_len,min_len


def entropy(codes, options, lang_state_embeds):
    """Calculate entropy of options over each batch

    option_codes: [N, D]
    lang_state_embeds: [B, D]
    """
    with torch.no_grad():
        N, D = codes.shape
        lang_state_embeds = lang_state_embeds.reshape(-1, 1, D)


        embed = codes.t()
        flatten = rearrange(lang_state_embeds, '... d -> (...) d')
        


        distance = -(
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        cond_probs = torch.softmax(distance / 2, dim=1)

        # get marginal probabilities
        probs = cond_probs.mean(dim=0)

        entropy = (-torch.log2(probs) * probs).sum()

        # calculate conditional entropy with language
        # sum over options, and then take expectation over language
        cond_entropy = (-torch.log2(cond_probs) * cond_probs).sum(1).mean(0)
        return (entropy, cond_entropy)


def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and
             obs.gripper_open == demo[i - 1].gripper_open and
             demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped

def _keypoint_discovery(demon: Demo, stopping_delta=0.1) -> List[int]:
    episode_keypoints = []
    prev_gripper_open = demon[0].gripper_open
    stopped_buffer = 0

    for i, obs in enumerate(demon):
      stopped = _is_stopped(demon, i, obs, stopped_buffer, stopping_delta)
      stopped_buffer = 4 if stopped else stopped_buffer - 1
      # if change in gripper, or end of episode.
      last = i == (len(demon) - 1)
      if i != 0 and (obs.gripper_open != prev_gripper_open or
                      last or stopped):
          episode_keypoints.append(i)
      prev_gripper_open = obs.gripper_open
    if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
          episode_keypoints[-2]:
        episode_keypoints.pop(-2)
    print('Found %d keypoints.' % len(episode_keypoints), episode_keypoints)
    return episode_keypoints

def loss_weights(replay_sample, beta=1.0):
    loss_weights = 1.0
    if 'sampling_probabilities' in replay_sample:
        probs = replay_sample['sampling_probabilities']
        loss_weights = 1.0 / torch.sqrt(probs + 1e-10)
        loss_weights = (loss_weights / torch.max(loss_weights)) ** beta
    return loss_weights


def soft_updates(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def stack_on_channel(x):
    # expect (B, T, C, ...)
    return torch.cat(torch.split(x, 1, dim=1), dim=2).squeeze(1)


def normalize_quaternion(quat):
    return np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)



def check_gimbal_lock(pred_rot_and_grip, gt_rot_and_grip, resolution):
    pred_rot_and_grip_np = pred_rot_and_grip.detach().cpu().numpy()
    gt_rot_and_grip_np = gt_rot_and_grip.detach().cpu().numpy()

    pred_rot = discrete_euler_to_quaternion(pred_rot_and_grip_np[:,:3], resolution)
    gt_rot = discrete_euler_to_quaternion(gt_rot_and_grip_np[:,:3], resolution)
    gimbal_lock_matches = [np.all(np.abs(pred_rot[i] - gt_rot[i]) < 1e-10) and
                           np.any(pred_rot_and_grip_np[i,:3] != gt_rot_and_grip_np[i, :3])
                           for i in range(pred_rot.shape[0])]
    return 0

def quaternion_to_discrete_euler(quaternion, resolution):
    euler = Rotation.from_quat(quaternion).as_euler('xyz', degrees=True) + 180
    assert np.min(euler) >= 0 and np.max(euler) <= 360
    disc = np.around((euler / resolution)).astype(int)
    disc[disc == int(360 / resolution)] = 0
    return disc


def discrete_euler_to_quaternion(discrete_euler, resolution):
    euluer = (discrete_euler * resolution) - 180
    return Rotation.from_euler('xyz', euluer, degrees=True).as_quat()

def point_to_voxel_index(
        point: np.ndarray,
        voxel_size: np.ndarray,
        coord_bounds: np.ndarray):
    bb_mins = np.array(coord_bounds[0:3])
    bb_maxs = np.array(coord_bounds[3:])
    dims_m_one = np.array([voxel_size] * 3) - 1
    bb_ranges = bb_maxs - bb_mins
    res = bb_ranges / (np.array([voxel_size] * 3) + 1e-12)
    voxel_indicy = np.minimum(
        np.floor((point - bb_mins) / (res + 1e-12)).astype(
            np.int32), dims_m_one)
    return voxel_indicy

def voxel_index_to_point(
        voxel_index: torch.Tensor,
        voxel_size: int,
        coord_bounds: np.ndarray):
    res = (coord_bounds[:, 3:] - coord_bounds[:, :3]) / voxel_size
    points = (voxel_index * res) + coord_bounds[:, :3]
    return points

def point_to_pixel_index(
        point: np.ndarray,
        extrinsics: np.ndarray,
        intrinsics: np.ndarray):
    point = np.array([point[0], point[1], point[2], 1])
    world_to_cam = np.linalg.inv(extrinsics)
    point_in_cam_frame = world_to_cam.dot(point)
    px, py, pz = point_in_cam_frame[:3]
    px = 2 * intrinsics[0, 2] - int(-intrinsics[0, 0] * (px / pz) + intrinsics[0, 2])
    py = 2 * intrinsics[1, 2] - int(-intrinsics[1, 1] * (py / pz) + intrinsics[1, 2])
    return px, py


def _compute_initial_camera_pose(scene):
    # Adapted from:
    # https://github.com/mmatl/pyrender/blob/master/pyrender/viewer.py#L1032
    centroid = scene.centroid
    scale = scene.scale
    if scale == 0.0:
        scale = DEFAULT_SCENE_SCALE
    s2 = 1.0 / np.sqrt(2.0)
    cp = np.eye(4)
    cp[:3, :3] = np.array([[0.0, -s2, s2], [1.0, 0.0, 0.0], [0.0, s2, s2]])
    hfov = np.pi / 6.0
    dist = scale / (2.0 * np.tan(hfov))
    cp[:3, 3] = dist * np.array([1.0, 0.0, 1.0]) + centroid
    return cp

def _from_trimesh_scene(
        trimesh_scene, bg_color=None, ambient_light=None):
    # convert trimesh geometries to pyrender geometries
    geometries = {name: pyrender.Mesh.from_trimesh(geom, smooth=False)
                  for name, geom in trimesh_scene.geometry.items()}
    # create the pyrender scene object
    scene_pr = pyrender.Scene(bg_color=bg_color, ambient_light=ambient_light)
    # add every node with geometry to the pyrender scene
    for node in trimesh_scene.graph.nodes_geometry:
        pose, geom_name = trimesh_scene.graph[node]
        scene_pr.add(geometries[geom_name], pose=pose)
    return scene_pr


def _create_bounding_box(scene, voxel_size, res):
    l = voxel_size * res
    T = np.eye(4)
    w = 0.01
    for trans in [[0, 0, l / 2], [0, l, l / 2], [l, l, l / 2], [l, 0, l / 2]]:
        T[:3, 3] = np.array(trans) - voxel_size / 2
        scene.add_geometry(trimesh.creation.box(
            [w, w, l], T, face_colors=[0, 0, 0, 255]))
    for trans in [[l / 2, 0, 0], [l / 2, 0, l], [l / 2, l, 0], [l / 2, l, l]]:
        T[:3, 3] = np.array(trans) - voxel_size / 2
        scene.add_geometry(trimesh.creation.box(
            [l, w, w], T, face_colors=[0, 0, 0, 255]))
    for trans in [[0, l / 2, 0], [0, l / 2, l], [l, l / 2, 0], [l, l / 2, l]]:
        T[:3, 3] = np.array(trans) - voxel_size / 2
        scene.add_geometry(trimesh.creation.box(
            [w, l, w], T, face_colors=[0, 0, 0, 255]))


def create_voxel_scene(
        voxel_grid: np.ndarray,
        q_attention: np.ndarray = None,
        highlight_coordinate: np.ndarray = None,
        highlight_gt_coordinate: np.ndarray = None,
        highlight_alpha: float = 1.0,
        voxel_size: float = 0.1,
        show_bb: bool = False,
        alpha: float = 0.5):
    _, d, h, w = voxel_grid.shape
    v = voxel_grid.transpose((1, 2, 3, 0))
    occupancy = v[:, :, :, -1] != 0
    alpha = np.expand_dims(np.full_like(occupancy, alpha, dtype=np.float32), -1)
    rgb = np.concatenate([(v[:, :, :, 3:6] + 1)/ 2.0, alpha], axis=-1)

    if q_attention is not None:
        q = np.max(q_attention, 0)
        q = q / np.max(q)
        show_q = (q > 0.75)
        occupancy = (show_q + occupancy).astype(bool)
        q = np.expand_dims(q - 0.5, -1)  # Max q can be is 0.9
        q_rgb = np.concatenate([
            q, np.zeros_like(q), np.zeros_like(q),
            np.clip(q, 0, 1)], axis=-1)
        rgb = np.where(np.expand_dims(show_q, -1), q_rgb, rgb)

    if highlight_coordinate is not None:
        x, y, z = highlight_coordinate
        occupancy[x, y, z] = True
        rgb[x, y, z] = [1.0, 0.0, 0.0, highlight_alpha]

    if highlight_gt_coordinate is not None:
        x, y, z = highlight_gt_coordinate
        occupancy[x, y, z] = True
        rgb[x, y, z] = [0.0, 0.0, 1.0, highlight_alpha]

    transform = trimesh.transformations.scale_and_translate(
        scale=voxel_size, translate=(0.0, 0.0, 0.0))
    trimesh_voxel_grid = trimesh.voxel.VoxelGrid(
        encoding=occupancy, transform=transform)
    geometry = trimesh_voxel_grid.as_boxes(colors=rgb)
    scene = trimesh.Scene()
    scene.add_geometry(geometry)
    if show_bb:
        assert d == h == w
        _create_bounding_box(scene, voxel_size, d)
    return scene


def visualise_voxel(voxel_grid: np.ndarray,
                    q_attention: np.ndarray = None,
                    highlight_coordinate: np.ndarray = None,
                    highlight_gt_coordinate: np.ndarray = None,
                    highlight_alpha: float = 1.0,
                    rotation_amount: float = 0.0,
                    show: bool = False,
                    voxel_size: float = 0.1,
                    offscreen_renderer: pyrender.OffscreenRenderer = None,
                    show_bb: bool = False,
                    alpha: float = 0.5):
    scene = create_voxel_scene(
        voxel_grid, q_attention, highlight_coordinate, highlight_gt_coordinate,
        highlight_alpha, voxel_size,
        show_bb, alpha)
    if show:
        scene.show()
    else:
        r = offscreen_renderer or pyrender.OffscreenRenderer(
            viewport_width=640, viewport_height=480, point_size=1.0)
        s = _from_trimesh_scene(
            scene, ambient_light=[0.8, 0.8, 0.8],
            bg_color=[1.0, 1.0, 1.0])
        cam = pyrender.PerspectiveCamera(
            yfov=np.pi / 4.0, aspectRatio=r.viewport_width/r.viewport_height)
        p = _compute_initial_camera_pose(s)
        t = Trackball(p, (r.viewport_width, r.viewport_height), s.scale, s.centroid)
        t.rotate(rotation_amount, np.array([0.0, 0.0, 1.0]))
        s.add(cam, pose=t.pose)
        color, depth = r.render(s)
        return color.copy()


def preprocess(img, dist='transporter'):
    """Pre-process input (subtract mean, divide by std)."""

    transporter_color_mean = [0.18877631, 0.18877631, 0.18877631]
    transporter_color_std = [0.07276466, 0.07276466, 0.07276466]
    transporter_depth_mean = 0.00509261
    transporter_depth_std = 0.00903967

    franka_color_mean = [0.622291933, 0.628313992, 0.623031488]
    franka_color_std = [0.168154213, 0.17626014, 0.184527364]
    franka_depth_mean = 0.872146842
    franka_depth_std = 0.195743116

    clip_color_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_color_std = [0.26862954, 0.26130258, 0.27577711]

    # choose distribution
    if dist == 'clip':
        color_mean = clip_color_mean
        color_std = clip_color_std
    elif dist == 'franka':
        color_mean = franka_color_mean
        color_std = franka_color_std
    else:
        color_mean = transporter_color_mean
        color_std = transporter_color_std

    if dist == 'franka':
        depth_mean = franka_depth_mean
        depth_std = franka_depth_std
    else:
        depth_mean = transporter_depth_mean
        depth_std = transporter_depth_std

    # convert to pytorch tensor (if required)
    if type(img) == torch.Tensor:
        def cast_shape(stat, img):
            tensor = torch.from_numpy(np.array(stat)).to(device=img.device, dtype=img.dtype)
            tensor = tensor.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            tensor = tensor.repeat(img.shape[0], 1, img.shape[-2], img.shape[-1])
            return tensor

        color_mean = cast_shape(color_mean, img)
        color_std = cast_shape(color_std, img)
        depth_mean = cast_shape(depth_mean, img)
        depth_std = cast_shape(depth_std, img)

        # normalize
        img = img.clone()
        img[:, :3, :, :] = ((img[:, :3, :, :] / 255 - color_mean) / color_std)
        img[:, 3:, :, :] = ((img[:, 3:, :, :] - depth_mean) / depth_std)
    else:
        # normalize
        img[:, :, :3] = (img[:, :, :3] / 255 - color_mean) / color_std
        img[:, :, 3:] = (img[:, :, 3:] - depth_mean) / depth_std
    return img


def rand_dist(size, min=-1.0, max=1.0):
    return (max-min) * torch.rand(size) + min


def rand_discrete(size, min=0, max=1):
    if min == max:
        return torch.zeros(size)
    return torch.randint(min, max+1, size)


def split_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def extract_obs(obs: Observation,
                t: int = 0,
                channels_last: bool = False,
                episode_length: int = 25):
    obs.joint_velocities = None
    grip_mat = obs.gripper_matrix
    grip_pose = obs.gripper_pose
    joint_pos = obs.joint_positions
    obs.gripper_pose = None
    obs.gripper_matrix = None
    obs.wrist_camera_matrix = None
    obs.joint_positions = None
    if obs.gripper_joint_positions is not None:
        obs.gripper_joint_positions = np.clip(
            obs.gripper_joint_positions, 0., 0.04)

    obs_dict = vars(obs)
    obs_dict = {k: v for k, v in obs_dict.items() if v is not None}
    robot_state = np.array([
        obs.gripper_open,
        *obs.gripper_joint_positions])
    # remove low-level proprioception variables that are not needed
    obs_dict = {k: v for k, v in obs_dict.items()
                if k not in REMOVE_KEYS}
    if not channels_last:
        # swap channels from last dim to 1st dim
        obs_dict = {k: np.transpose(
            v, [2, 0, 1]) if v.ndim == 3 else np.expand_dims(v, 0)
                    for k, v in obs_dict.items() if type(v) == np.ndarray or type(v) == list}
    else:
        # add extra dim to depth data
        obs_dict = {k: v if v.ndim == 3 else np.expand_dims(v, -1)
                    for k, v in obs_dict.items()}
    obs_dict['robot_state'] = np.array(robot_state, dtype=np.float32)

    # binary variable indicating if collisions are allowed or not while planning paths to reach poses
    if hasattr(obs,'ignore_collisions'):
        obs_dict['ignore_collisions'] = np.array([obs.ignore_collisions], dtype=np.float32)
    else:
        obs_dict['ignore_collisions'] = np.array([1.0],dtype=np.float32)

    for (k, v) in [(k, v) for k, v in obs_dict.items() if 'point_cloud' in k]:
        obs_dict[k] = v.astype(np.float32)

    # add timestep to low_dim_state
    time = (1. - (t / float(episode_length - 1))) * 2. - 1.
    obs_dict['robot_state'] = np.concatenate([obs_dict['robot_state'], [time]]).astype(np.float32)

    obs.gripper_matrix = grip_mat
    obs.joint_positions = joint_pos
    obs.gripper_pose = grip_pose

    return obs_dict


def pad(x, max_len, axis=1, const=0, mode='post'):
    """Pads input sequence with given const along a specified dim

    Inputs:
        x: Sequence to be padded
        max_len: Max padding length
        axis: Axis to pad (Default: 1)
        const: Constant to pad with (Default: 0)
        mode: ['pre', 'post'] Specifies whether to add padding pre or post to the sequence
    """

    if isinstance(x, tuple):
        x = np.array(x)

    pad_size = max_len - x.shape[axis]
    if pad_size <= 0:
        return x

    npad = [(0, 0)] * x.ndim
    if mode == 'pre':
        npad[axis] = (pad_size, 0)
    elif mode == 'post':
        npad[axis] = (0, pad_size)
    else:
        raise NotImplementedError

    if isinstance(x, np.ndarray):
        x_padded = np.pad(x, pad_width=npad, mode='constant', constant_values=const)
    elif isinstance(x, torch.Tensor):
        # pytorch starts padding from final dim so need to reverse chaining order
        npad = tuple(itertools.chain(*reversed(npad)))
        x_padded = F.pad(x, npad, mode='constant', value=const)
    else:
        raise NotImplementedError
    return x_padded

def create_obs_config(camera_names: List[str],
                       camera_resolution: List[int]):
    unused_cams = CameraConfig()
    unused_cams.set_all(False)
    used_cams = CameraConfig(
        rgb=True,
        point_cloud=True,
        mask=False,
        depth=False,
        image_size=camera_resolution,
        render_mode=RenderMode.OPENGL)

    cam_obs = []
    kwargs = {}
    for n in camera_names:
        kwargs[n] = used_cams
        cam_obs.append('%s_rgb' % n)
        cam_obs.append('%s_pointcloud' % n)

    # Some of these obs are only used for keypoint detection.
    obs_config = ObservationConfig(
        front_camera=kwargs.get('front', unused_cams),
        left_shoulder_camera=kwargs.get('left_shoulder', unused_cams),
        right_shoulder_camera=kwargs.get('right_shoulder', unused_cams),
        wrist_camera=kwargs.get('wrist', unused_cams),
        overhead_camera=kwargs.get('overhead', unused_cams),
        joint_forces=False,
        joint_positions=True,
        joint_velocities=True,
        task_low_dim_state=False,
        gripper_touch_forces=False,
        gripper_pose=True,
        gripper_open=True,
        gripper_matrix=True,
        gripper_joint_positions=True,
    )
    return obs_config


def get_device(gpu):
    if gpu is not None and gpu >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:%d" % gpu)
        torch.backends.cudnn.enabled = torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    return device
