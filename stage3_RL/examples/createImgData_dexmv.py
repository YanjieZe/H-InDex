# Copyright (c) Rutav Shah, Indian Institute of Technlogy Kharagpur
# Copyright (c) Facebook, Inc. and its affiliates

import mj_envs
import click
import gym
from pathlib import Path
import pickle
home = str(Path.home())
from mjrl.utils.gym_env import GymEnv
from hand_imitation.env.environments.ycb_relocate_env import YCBRelocate
from hand_imitation.env.environments.mug_pour_water_env import WaterPouringEnv
from hand_imitation.env.environments.mug_place_object_env import MugPlaceObjectEnv
import numpy as np
import os
import rrl
import torchvision, torch
import cv2

from termcolor import colored, cprint

_dexmv_envs = {'pour-v0'}

DESC = '''
Helper script to create demos.\n
USAGE:\n
    Create demos on the env\n
    $ \n
'''

def render(env, img_size, cam):
	img = env.sim.render(mode='offscreen', width=img_size, height=img_size, camera_name=cam, device_id=0)
	# flip
	img = img[::-1, :, :]
	return img
    	
seed = 123
@click.command(help=DESC)
@click.option('--data_dir', type=str, help='Directory to save data', required=True)
@click.option('--env_name', type=str, help='environment to load', required= True)
@click.option('--num_demos', type=int, help='Number of demonstrations', default=25)
@click.option('--mode', type=str, help='Mode : evaluation, exploration', default="exploration")
@click.option('--cam', type=str, help='Camera : frontview, backview, sideview', default="frontview")
def main(data_dir, env_name, num_demos, mode, cam):
	print("Data Directory : ", data_dir)
	base_dir = list(set(rrl.__path__))[0] + "/"

	env_name, object_name = env_name.split("-") if "-" in env_name else (env_name, None)

	# create policy
	if env_name == "relocate":
		policy = f"../../third_party/dexmv-sim/pretrained_model/{env_name}-{object_name}.pickle"
	else:
		policy = f"../../third_party/dexmv-sim/pretrained_model/{env_name}.pickle"
	print("Policy : ", policy)
	pi = pickle.load(open(policy, 'rb'))

	available_cameras = ['frontview', 'backview', 'sideview']
	# create environment
	friction = (1, 0.5, 0.01)
	if env_name == "relocate":
		if object_name is None:
			raise ValueError("For relocate task, object name is needed.")
		e = YCBRelocate(has_renderer=False, object_name=object_name, friction=friction, object_scale=0.8,
						solref="-6000 -300", randomness_scale=0.25)
		T = 100
	elif env_name == "pour":
		e = WaterPouringEnv(has_renderer=False, scale=1.0, tank_size=(0.15, 0.15, 0.08))
		T = 200
	elif env_name == "place_inside":
		e = MugPlaceObjectEnv(has_renderer=False, object_scale=0.8, mug_scale=1.5)
		T = 200
	else:
		raise NotImplementedError

	cam_name = cam
	e.set_seed(seed)
	demo_paths = []
	for data_id in range(num_demos):
		obs = e.reset()
		obs_img = render(env=e, img_size=256, cam=cam_name)
		done = False
		new_path = {}
		new_path_obs = [obs]
		new_path_obs_img = [obs_img]
		new_path_actions = []
		new_path_rewards = []
		ep_reward = 0
		step = 0
		while not done:
			action = pi.get_action(obs)[0] if mode == 'exploration' else pi.get_action(obs)[1]['evaluation']
			next_obs, reward, done, info = e.step(action)
			obs_img = render(env=e, img_size=256, cam=cam_name)
			ep_reward += reward

			new_path_obs.append(next_obs)
			new_path_obs_img.append(obs_img)
			new_path_actions.append(action)
			new_path_rewards.append(reward)


			obs = next_obs
			step += 1
			if step >= T:
				break
		print("reward : ", ep_reward)

		video_dir = os.path.join(data_dir, str(data_id))
		if not os.path.exists(video_dir):
			os.makedirs(video_dir)
		
		for img_id in range(len(new_path_obs_img)):
			img = new_path_obs_img[img_id]
			img_path = os.path.join(video_dir, str(img_id) + ".png")
			# convert to BGR
			img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
			cv2.imwrite(img_path, img)

		print("Dumping video demos at : ", video_dir)

		
if __name__ == "__main__":
	main()
