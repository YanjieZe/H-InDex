# Copyright (c) Rutav Shah, Indian Institute of Technlogy Kharagpur
# Copyright (c) Facebook, Inc. and its affiliates

from mjrl.utils.gym_env import GymEnv

from hand_imitation.env.environments.mug_place_object_env import MugPlaceObjectEnv
from hand_imitation.env.environments.mug_pour_water_env import WaterPouringEnv
from hand_imitation.env.environments.ycb_relocate_env import YCBRelocate

    
from rrl.multicam import RRL, RRL_dexmv
import os
from termcolor import colored
import torch
import torch.nn as nn
import torchvision
import numpy as np

class Spec:
    def __init__(self, env=None, env_name="relocate-mug-1"):
        self.observation_dim = env.reset().shape[0]
        self.action_dim = env.action_spec[0].shape[0]
        self.env_id = env_name


dexmv_envs = ['pour-v0', 'place_inside-v0', 'relocate-mug-v0',  'relocate-foam_brick-v0', 
              'relocate-large_clamp-v0', 'relocate-mustard_bottle-v0', 'relocate-potted_meat_can-v0',
              'relocate-potted_meat_can-v0', 'relocate-sugar_box-v0','relocate-tomato_soup_can-v0',]

def make_env(env, cam_list=[], encoder=None, from_pixels=False, encoder_type=None, hybrid_state=None, episode_length=None, arena_id=0) :
    if env == 'pour-v0':
        e = WaterPouringEnv(has_renderer=False, scale=1.0, tank_size=(0.15, 0.15, 0.08), arena_id=arena_id)
    elif env == 'place_inside-v0':
        e = MugPlaceObjectEnv(has_renderer=False, object_scale=0.8, mug_scale=1.5, arena_id=arena_id)
    elif env in ['relocate-v0', 'hammer-v0', 'pen-v0', 'door-v0']:
        e = GymEnv(env)
    elif 'relocate' in env:
        env_name, object_name, _ = env.split('-')
        if object_name is None:
            raise ValueError("For relocate task, object name is needed.")
        friction = (1, 0.5, 0.01)
        e = YCBRelocate(has_renderer=False, object_name=object_name, friction=friction, object_scale=0.8,
                        solref="-6000 -300", randomness_scale=0.25, arena_id=arena_id)
    else:
        raise ValueError(f"Invalid environment name: {env}")

    env_kwargs = None
    if from_pixels :
        height = 84
        width = 84
        latent_dim = height*width*len(cam_list)*3

    if encoder_type and encoder_type == 'resnet34':
        assert from_pixels==True
        height = 256
        width = 256
        latent_dim = 512*len(cam_list)

    if encoder_type and encoder_type != 'resnet34':
        assert from_pixels==True
        height = 256
        width = 256
        latent_dim = encoder.latent_dim*len(cam_list)
        print(colored("[make_env] latent_dim: {}".format(latent_dim), 'green'))


    if from_pixels:
        if env in dexmv_envs:
            e = RRL_dexmv(e, env_id=env, cameras=cam_list, encoder_type=encoder_type, encoder=encoder,
            height=height, width=width, latent_dim=latent_dim, hybrid_state=hybrid_state, episode_length=episode_length)
        elif env in ['relocate-v0', 'hammer-v0', 'pen-v0', 'door-v0']:
            e = RRL(e, cameras=cam_list, encoder_type=encoder_type, encoder=encoder,
                height=height, width=width, latent_dim=latent_dim, hybrid_state=hybrid_state)
        else:
            raise ValueError(f"Invalid environment name: {env}")
        env_kwargs = {'rrl_kwargs' : e.env_kwargs}
    return e, env_kwargs

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def preprocess_args(args):
	job_data = {}
	job_data['seed'] = args.seed
	job_data['env'] = args.env
	job_data['output'] = args.output
	job_data['from_pixels'] = args.from_pixels
	job_data['hybrid_state'] = args.hybrid_state
	job_data['stack_frames'] = args.stack_frames
	job_data['encoder_type'] = args.encoder_type
	job_data['encoder_ckpt'] = args.encoder_ckpt
	job_data['cam1'] = args.cam1
	job_data['cam2'] = args.cam2
	job_data['cam3'] = args.cam3
	job_data['algorithm'] = args.algorithm
	job_data['num_cpu'] = args.num_cpu
	job_data['save_freq'] = args.save_freq
	job_data['eval_rollouts'] = args.eval_rollouts
	job_data['demo_file'] = args.demo_file
	job_data['bc_batch_size'] = args.bc_batch_size
	job_data['bc_epochs'] = args.bc_epochs
	job_data['bc_learn_rate'] = args.bc_learn_rate
	#job_data['policy_size'] = args.policy_size
	job_data['policy_size'] = tuple(map(int, args.policy_size.split(', ')))
	job_data['vf_batch_size'] = args.vf_batch_size
	job_data['vf_epochs'] = args.vf_epochs
	job_data['vf_learn_rate'] = args.vf_learn_rate
	job_data['rl_step_size'] = args.rl_step_size
	job_data['rl_gamma'] = args.rl_gamma
	job_data['rl_gae'] = args.rl_gae
	job_data['rl_num_traj'] = args.rl_num_traj
	job_data['rl_num_iter'] = args.rl_num_iter
	job_data['lam_0'] = args.lam_0
	job_data['lam_1'] = args.lam_1
	print("==========================================================")
	print(job_data)
	print("==========================================================")
	
	return job_data



def PSNR(img1, img2, max_val=1):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = max_val
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


class VGG16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 14):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(14, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 28):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        output = [X]
        h = self.slice1(X)
        output.append(h)
        h = self.slice2(h)
        output.append(h)
        h = self.slice3(h)
        output.append(h)
        h = self.slice4(h)
        output.append(h)
        h = self.slice5(h)
        output.append(h)
        return output


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg16 = VGG16(requires_grad=False)
        self.vgg16.eval()
        self.criterion = nn.MSELoss(reduction='mean')
        self.loss_mean = [0.1966, 0.8725, 3.4260, 7.4396, 4.1430, 1.1304]
        self.momentum = 0.01

    def forward(self, images, pred_images):
        output_images = self.vgg16(images)
        output_pred = self.vgg16(pred_images)
        loss = []
        for i in range(len(output_images)):
            l = self.criterion(output_images[i], output_pred[i])
            l = l.mean()
            self.loss_mean[i] = self.loss_mean[i] + \
                self.momentum * (l.detach() - self.loss_mean[i])
            l = l / self.loss_mean[i]
            loss.append(l)
        loss = torch.stack(loss).sum()
        return loss