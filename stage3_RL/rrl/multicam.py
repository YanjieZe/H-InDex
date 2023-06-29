# Copyright (c) Rutav Shah, Indian Institute of Technlogy Kharagpur
# Copyright (c) Facebook, Inc. and its affiliates

import gym
from abc import ABC
import numpy as np
from rrl.encoder import Encoder, IdentityEncoder, ResNetFT, EncoderBN
from rrl.representations import TTPEncoder, TTPHumanEncoder
from rrl.representations import TTPResnetEncoder

from rrl.representations import TimeCorrEncoder



from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import time
from termcolor import colored, cprint

_mj_envs = {'pen-v0', 'hammer-v0', 'door-v0', 'relocate-v0'}
_dexmv_envs = {'pour-v0', 'place_inside-v0'}


def make_encoder(encoder_type, device, is_eval=True, ckpt_path=None, test_time_momentum=0.0) :

    if encoder_type in ['resnet34', 'resnet50'] :
        encoder = Encoder(encoder_type)
    elif encoder_type in ['resnet50_bn']:
        encoder = EncoderBN('resnet50', bn_momentum=test_time_momentum)
    elif encoder_type == 'identity' :
        encoder = IdentityEncoder()
    elif encoder_type in ['ttp', 'ttp_frankmocap_hand']:
        cfg_path = "../rrl/representations/TTP/imm_joint_model.yaml"
        encoder = TTPEncoder(cfg_path=cfg_path, ckpt_path=ckpt_path)
    elif encoder_type == "ttp_frankmocap_hand_bn":
        from rrl.representations.ttp_encoder import TTPEncoderBN
        cfg_path = "../rrl/representations/TTP/imm_joint_model.yaml"
        encoder = TTPEncoderBN(cfg_path=cfg_path, ckpt_path=ckpt_path, bn_momentum=test_time_momentum)
    elif encoder_type == "ttp_human":
        cfg_path = "../rrl/representations/TTP/imm_joint_model.yaml"
        encoder = TTPHumanEncoder(cfg_path=cfg_path, ckpt_path=ckpt_path)
    elif encoder_type == "rrl":
        encoder = Encoder('resnet50')
    elif encoder_type == 'dino' :
        from rrl.representations import DINOEncoder
        encoder = DINOEncoder(ckpt_path=ckpt_path)
    elif encoder_type == "pvr":
        from rrl.representations import MoCoEncoder
        encoder = MoCoEncoder(ckpt_path=ckpt_path)
    elif encoder_type == "r3m":
        from rrl.representations import R3MEncoder
        model_and_config_folder = ckpt_path
        encoder = R3MEncoder(model_and_config_folder=model_and_config_folder)
    elif encoder_type == "mvp":
        from rrl.representations import MVPEncoder
        encoder = MVPEncoder(ckpt_path=ckpt_path)
    elif encoder_type == "vc1":
        from rrl.representations import VC1Encoder
        encoder = VC1Encoder(ckpt_path=ckpt_path)
    elif encoder_type == "alphapose":
        from rrl.representations import AlphaPoseEncoder
        encoder = AlphaPoseEncoder(ckpt_path=ckpt_path)
    elif encoder_type == "frankmocap_hand":
        from rrl.encoder import FrankMocapHandEncoder
        encoder = FrankMocapHandEncoder("resnet50", ckpt_path)
    elif encoder_type == "frankmocap_hand_bn":
        from rrl.encoder import FrankMocapHandEncoderBN
        encoder = FrankMocapHandEncoderBN("resnet50", ckpt_path, bn_momentum=test_time_momentum)
    elif encoder_type == "frankmocap_hand_fusion":
        from rrl.encoder import FrankMocapHandFusionEncoder
        encoder = FrankMocapHandFusionEncoder("resnet50", ckpt_path)
    elif encoder_type == "frankmocap_hand_joint":
        from rrl.encoder import FrankMocapHandJointEncoder
        encoder = FrankMocapHandJointEncoder("resnet50", ckpt_path)
    elif encoder_type == "frankmocap_hand_state":
        from rrl.encoder import FrankMocapHandStateEncoder
        encoder = FrankMocapHandStateEncoder("resnet34", ckpt_path)
    elif encoder_type == "frankmocap_body":
        from rrl.encoder import FrankMocapBodyEncoder
        encoder = FrankMocapBodyEncoder("resnet50", ckpt_path)
    elif encoder_type == "hand_object_detector":
        from rrl.representations.hand_object_encoder import HandObjectDetectExtractor
        encoder = HandObjectDetectExtractor(ckpt_path)
    else:
        raise Exception(f"Encoder type {encoder_type} not supported")
    if is_eval:
        encoder.eval()
    encoder.to(device)
    return encoder

class RRL(gym.Env, ABC):
    def __init__(self, env, cameras, encoder_type="resnet34", encoder=None, latent_dim=512, hybrid_state=True, channels_first=False, height=100, width=100, device_id=0):
        num_gpu = torch.cuda.device_count()
        device_id = device_id % num_gpu
        
        self._env = env
        self.env = env
        self.env_id = env.env.unwrapped.spec.id

        self.cameras = cameras
        self.encoder_type = encoder_type
        self.latent_dim = latent_dim

        self.hybrid_state = hybrid_state
        self.channels_first = channels_first
        self.height = height
        self.width = width
        self.action_space = self._env.action_space
        self.device_id = device_id
        print(colored(f"[rendering device] Using device {device_id}", "green"))
        self.env_kwargs = {'cameras' : cameras, 'encoder_type': encoder_type, 'encoder': encoder, 'latent_dim' : latent_dim, 'hybrid_state': hybrid_state, 'channels_first' : channels_first, 'height' : height, 'width' : width}

        shape = [latent_dim]
        self._observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )
        self.sim = env.env.sim

        self._env.spec.observation_dim = latent_dim
        if hybrid_state :
            self._env.spec.observation_dim += 24 # Assuming 24 states for adroit hand.

        self.device = "cuda:"+str(device_id)
        self.encoder = encoder
        self.transforms = self.encoder.get_transform()
        self.spec = self._env.spec
        self.observation_dim = self.spec.observation_dim
        self.horizon = self._env.env.spec.max_episode_steps


    def get_obs(self, state=None, img_size=None):

        env_state = self._env.env.get_env_state()
        qp = env_state['qpos']

        if self.env_id == 'pen-v0':
            qp = qp[:-6]
        elif self.env_id == 'door-v0':
            qp = qp[4:-2]
        elif self.env_id == 'hammer-v0':
            qp = qp[2:-7]
        elif self.env_id == 'relocate-v0':
            qp = qp[6:-6]
        
        imgs = []
        for cam in self.cameras :
            if img_size is None :
                img = self._env.env.sim.render(width=self.width, height=self.height, mode='offscreen', camera_name=cam, device_id=self.device_id)
            else :
                img = self._env.env.sim.render(width=img_size, height=img_size, mode='offscreen', camera_name=cam, device_id=self.device_id)
            img = img[::-1, :, : ].transpose((2, 0, 1)) # Image given has to be flipped
            img = np.ascontiguousarray(img, dtype=np.float32)
            imgs.append(img)

        ## put img on gpu and then do transform !! this saves cpu
        imgs = np.array(imgs)
        inp_img = self.transforms(torch.from_numpy(imgs/ 255.).to(self.device)) # [num_cam, C, H, W]

        z = self.encoder.get_features(inp_img).reshape(-1)
        if self.encoder_type == "identity":
            z = z.cpu().numpy()
        # assert z.shape[0] == self.latent_dim, "Encoded feature length : {}, Expected : {}".format(z.shape[0], self.latent_dim)
        if self.hybrid_state:
            z = np.hstack((z, qp))
        return z, imgs

    def get_env_infos(self):
        return self._env.get_env_infos()

    def set_seed(self, seed):
        return self._env.set_seed(seed)

    def reset(self):
        obs = self._env.reset()
        obs, img_obs = self.get_obs(obs)
        return obs, img_obs

    def step(self, action):
        obs, reward, done, env_info = self._env.step(action)
        obs, img_obs = self.get_obs(obs)
        return obs, img_obs, reward, done, env_info

    def set_env_state(self, state):
        return self._env.set_env_state(state)

    def get_env_state(self):
        return self._env.get_env_state()

    def evaluate_policy(self, policy,
    					num_episodes=5,
    					horizon=None,
    					gamma=1,
    					visual=False,
    					percentile=[],
    					get_full_dist=False,
    					mean_action=False,
    					init_env_state=None,
    					terminate_at_done=True,
    					seed=123):

        self.set_seed(seed)
        horizon = self.horizon if horizon is None else horizon
        mean_eval, std, min_eval, max_eval = 0.0, 0.0, -1e8, -1e8
        ep_returns = np.zeros(num_episodes)
        num_success =[]
        self.encoder.eval()

        for ep in range(num_episodes):
            o, img_o = self.reset()
            success = 0.
            if init_env_state is not None:
                self.set_env_state(init_env_state)
            t, done = 0, False
            while t < horizon and (done == False or terminate_at_done == False):
                self.render() if visual is True else None
                o, img_o = self.get_obs(self._env.get_obs())
                a = policy.get_action(o)[1]['evaluation'] if mean_action is True else policy.get_action(o)[0]
                o, img_o, r, done, info = self.step(a)
                ep_returns[ep] += (gamma ** t) * r
                success += float(info['goal_achieved'])
                t += 1
            is_success = success > 25.
            num_success.append(is_success)

        mean_eval, std = np.mean(ep_returns), np.std(ep_returns)
        min_eval, max_eval = np.amin(ep_returns), np.amax(ep_returns)
        all_data = {}
        all_data['return_mean'], all_data['return_std'], all_data['return_min'], all_data['return_max'] = mean_eval, std, min_eval, max_eval

        SR = np.mean(num_success)
        all_data['success_rate'] = SR

        percentile_stats = []
        for p in percentile:
            percentile_stats.append(np.percentile(ep_returns, p))

        full_dist = ep_returns if get_full_dist is True else None
        all_data['percentile'] = percentile_stats
        all_data['full_dist'] = full_dist
        
        return all_data

class RRL_dexmv(gym.Env, ABC):
    def __init__(self, env, env_id,  cameras, encoder_type="resnet34", encoder=None, latent_dim=512, \
                 hybrid_state=True, channels_first=False, height=100, width=100, device_id=0, episode_length=100):
        num_gpu = torch.cuda.device_count()
        device_id = device_id % num_gpu
        self._env = env
        self.env = env
        self.env_id = env_id

        self.cameras = cameras
        self.encoder_type = encoder_type
        self.latent_dim = latent_dim

        self.hybrid_state = hybrid_state
        self.channels_first = channels_first
        self.height = height
        self.width = width
        self.action_space =  env.action_spec[0]
        self.device_id = device_id
        self.env_kwargs = {'cameras' : cameras, 'encoder_type': encoder_type, 'encoder': encoder, 'latent_dim' : latent_dim, 'hybrid_state': hybrid_state, 'channels_first' : channels_first, 'height' : height, 'width' : width}

        shape = [latent_dim]
        self._observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )
        self.sim = env.sim

        self.observation_dim = latent_dim
        if hybrid_state :
            self.observation_dim += 30

        self._env.spec.observation_dim = self.observation_dim
        self.device = "cuda:"+str(device_id)
        self.encoder = encoder
        self.transforms = self.encoder.get_transform()
        self.spec = self._env.spec
        self.spec.observation_dim = self.observation_dim
        self.horizon = episode_length


    def get_obs(self, state=None, img_size=None):

        qp =  self.env.data.qpos.ravel()[:30]

        # if self.env_id == 'pen-v0':
        #     qp = qp[:-6]
        # elif self.env_id == 'door-v0':
        #     qp = qp[4:-2]
        # elif self.env_id == 'hammer-v0':
        #     qp = qp[2:-7]
        # elif self.env_id == 'relocate-v0':
        #     qp = qp[6:-6]
        
        imgs = []
        for cam in self.cameras :
            if img_size is None :
                img = self.sim.render(width=self.width, height=self.height, mode='offscreen', camera_name=cam, device_id=self.device_id)
            else :
                img = self.sim.render(width=img_size, height=img_size, mode='offscreen', camera_name=cam, device_id=self.device_id)
            img = img[::-1, :, : ].transpose((2, 0, 1)) # Image given has to be flipped
            img = np.ascontiguousarray(img, dtype=np.float32)
            imgs.append(img)

        ## put img on gpu and then do transform !! this saves cpu
        inp_img = self.transforms(torch.from_numpy(np.array(imgs) / 255.).to(self.device)) # [num_cam, C, H, W]

        z = self.encoder.get_features(inp_img).reshape(-1)
        if self.encoder_type == "identity":
            z = z.cpu().numpy()
        # assert z.shape[0] == self.latent_dim, "Encoded feature length : {}, Expected : {}".format(z.shape[0], self.latent_dim)
        if self.hybrid_state:
            z = np.hstack((z, qp))

        return z, inp_img

    def get_env_infos(self):
        return {}

    def set_seed(self, seed):
        return self._env.set_seed(seed)

    def reset(self):
        obs = self._env.reset()
        obs, img_obs = self.get_obs(obs)
        return obs, img_obs

    def step(self, action):
        obs, reward, done, env_info = self._env.step(action)
        obs, img_obs = self.get_obs(obs)
        env_info['goal_achieved'] = 0. # hardcode. we use reward to measure success later.
        return obs, img_obs, reward, done, env_info

    def set_env_state(self, state):
        qpos = state['qpos']
        qvel = state['qvel']
        return self._env.set_state(qpos, qvel)

    def get_env_state(self):
        return self._env.get_env_state()

    def evaluate_policy(self, policy,
    					num_episodes=5,
    					horizon=None,
    					gamma=1,
    					visual=False,
    					percentile=[],
    					get_full_dist=False,
    					mean_action=False,
    					init_env_state=None,
    					terminate_at_done=True,
    					seed=123):

        self.set_seed(seed)
        horizon = self.horizon if horizon is None else horizon
        mean_eval, std, min_eval, max_eval = 0.0, 0.0, -1e8, -1e8
        ep_returns = np.zeros(num_episodes)
        num_success =[]
        self.encoder.eval()

        for ep in range(num_episodes):
            o, img_o = self.reset()
            success = 0.
            if init_env_state is not None:
                self.set_env_state(init_env_state)
            t, done = 0, False
            while t < horizon and (done == False or terminate_at_done == False):
                self.render() if visual is True else None
                o, img_o = self.get_obs()
                a = policy.get_action(o)[1]['evaluation'] if mean_action is True else policy.get_action(o)[0]
                o, img_o, r, done, info = self.step(a)
                ep_returns[ep] += (gamma ** t) * r
                success += float(info['goal_achieved'])
                t += 1
            is_success = success > 25.
            num_success.append(is_success)

        mean_eval, std = np.mean(ep_returns), np.std(ep_returns)
        min_eval, max_eval = np.amin(ep_returns), np.amax(ep_returns)
        all_data = {}
        all_data['return_mean'], all_data['return_std'], all_data['return_min'], all_data['return_max'] = mean_eval, std, min_eval, max_eval

        SR = np.mean(num_success)
        all_data['success_rate'] = SR

        percentile_stats = []
        for p in percentile:
            percentile_stats.append(np.percentile(ep_returns, p))

        full_dist = ep_returns if get_full_dist is True else None
        all_data['percentile'] = percentile_stats
        all_data['full_dist'] = full_dist
        
        return all_data
