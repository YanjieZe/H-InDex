# Copyright (c) Rutav Shah, Indian Institute of Technlogy Kharagpur
# Copyright (c) Facebook, Inc. and its affiliates

import mj_envs
import click
import gym
from pathlib import Path
import pickle
home = str(Path.home())
from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
import mjrl
from mjrl.policies import *
import numpy as np
import os
import rrl
import cv2

_mj_envs = {'pen-v0', 'hammer-v0', 'door-v0', 'relocate-v0', 'tools-v0'}
_mjrl_envs = {'mjrl_peg_insertion-v0', 'mjrl_reacher_7dof-v0'}
DESC = '''
Helper script to create demos.\n
USAGE:\n
    Create demos on the env\n
    $ \n
'''
seed = 123

def render_obs(env, img_size=224, camera_name="vil_camera", device=0):
	img = env.env.sim.render(width=img_size, height=img_size, \
			mode='offscreen', camera_name=camera_name, device_id=device)
	img = img[::-1, :, : ] # Image given has to be flipped
	return img

@click.command(help=DESC)
@click.option('--data_dir', type=str, help='Directory to save data', required=True)
@click.option('--env_name', type=str, help='environment to load', required= True)
@click.option('--num_demos', type=int, help='Number of demonstrations', default=25)
@click.option('--mode', type=str, help='Mode : evaluation, exploration', default="exploration")
@click.option('--policy', type=str, help='Location to the policy', required=True)
@click.option('--img_size', type=int, help='Image size', default=256)
@click.option('--camera_name', type=str, help='Camera name', default="vil_camera")
@click.option('--gpu_id', type=int, help='GPU ID', default=0)
def main(data_dir, env_name, num_demos, mode, policy, img_size, camera_name, gpu_id):
	print("Data Directory : ", data_dir)
	print("Policy : ", policy)
	pi = pickle.load(open(policy, 'rb'))
	e = GymEnv(env_name)
	e.set_seed(seed)

	for data_id in range(num_demos):
		img_list = []
		obs = e.reset()

		img_obs = render_obs(e, img_size=img_size, camera_name=camera_name, device=gpu_id)
		img_list.append(img_obs)

		if env_name in _mj_envs or env_name in _mjrl_envs :
			init_state_dict = e.get_env_state()
		else:
			print("Please enter valid environment. Mentioned : ", env_name)
			exit()

		done = False
		new_path = {}
		ep_reward = 0
		step = 0
		while not done:
			action = pi.get_action(obs)[0] if mode == 'exploration' else pi.get_action(obs)[1]['evaluation']
			next_obs, reward, done, info = e.step(action)

			img_obs = render_obs(e, img_size=img_size, camera_name=camera_name, device=gpu_id)
			img_list.append(img_obs)

			ep_reward += reward

			obs = next_obs
			step += 1
		print("Episode Reward : ", ep_reward)

		video_dir = os.path.join(data_dir, str(data_id))
		if not os.path.exists(video_dir):
			os.makedirs(video_dir)

		for img_id in range(len(img_list)):
			img = img_list[img_id]
			img_path = os.path.join(video_dir, str(img_id) + ".png")
			# convert to BGR
			img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
			cv2.imwrite(img_path, img)

		print("Dumping video demos at : ", video_dir)

if __name__ == "__main__":
	main()
