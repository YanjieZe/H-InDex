# Copyright (c) Rutav Shah, Indian Institute of Technlogy Kharagpur
# Copyright (c) Facebook, Inc. and its affiliates

# python job_script.py --config dapg.txt --output dir_name --cam1 cam_name --cam2 cam_name --cam3 cam_name
"""
This is a job script for running NPG/DAPG on hand tasks and other gym envs.
Note that DAPG generalizes PG and BC init + PG finetuning.
With appropriate settings of parameters, we can recover the full family.
"""



from hand_imitation.env.environments.mug_place_object_env import MugPlaceObjectEnv
from hand_imitation.env.environments.mug_pour_water_env import WaterPouringEnv
from hand_imitation.env.environments.ycb_relocate_env import YCBRelocate

from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.algos.dapg import DAPG
from mjrl.algos.behavior_cloning import BC
from mjrl.utils.train_agent import train_agent
from mjrl.samplers.core import sample_paths
import os
import json
import mjrl.envs
import mj_envs
import time as timer
import pickle
import argparse
from pathlib import Path
import multiprocessing
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from rrl.utils import make_env, preprocess_args
from rrl.multicam import make_encoder
from rrl.adaptation import test_time_adaptation

from termcolor import colored, cprint
import torch
import numpy as np
from tqdm import tqdm
import wandb
import warnings
warnings.filterwarnings("ignore")

home = str(Path.home())

def render(env, img_size, cam):
	img = env.sim.render(mode='offscreen', width=img_size, height=img_size, camera_name=cam, device_id=0)
	# flip
	img = img[::-1, :, :]
	return img

# ===============================================================================
# Get command line arguments
# ===============================================================================

@hydra.main(config_name="hammer_dapg", config_path="config/dexmv")
def main(args : DictConfig):
    job_data = preprocess_args(args)

    if args.use_wandb:
        os.environ["WANDB_SILENT"] = "true"
        wandb.init(project=str(args.wandb_project), group=str(args.wandb_group), name=str(args.wandb_name))
        wandb.config.update(dict(args))
        print(colored(f"[wandb] init in project: {args.wandb_project}, group: {args.wandb_group}, name: {args.wandb_name}", "cyan"))
    
    assert 'algorithm' in job_data.keys()
    assert any([job_data['algorithm'] == a for a in ['NPG', 'BCRL', 'DAPG']])
    job_data['lam_0'] = 0.0 if 'lam_0' not in job_data.keys() else job_data['lam_0']
    job_data['lam_1'] = 0.0 if 'lam_1' not in job_data.keys() else job_data['lam_1']


    JOB_DIR = os.path.join("logs", job_data['output'], "seed{}".format(job_data['seed']) )
    if not os.path.exists(JOB_DIR):
        os.makedirs(JOB_DIR)
    print(colored(f"Job directory: {JOB_DIR}", "red"))
    EXP_FILE = JOB_DIR + '/job_config.json'
    with open(EXP_FILE, 'w') as f:
        json.dump(job_data, f, indent=4)

    # ===============================================================================
    # Train Loop
    # ===============================================================================

    if job_data['from_pixels'] == True :
        if args.cam1 is None:
            print("Please pass cameras in the arguments.")
            exit()

        encoder = make_encoder(encoder_type=job_data['encoder_type'], device='cuda', is_eval=True, ckpt_path=job_data['encoder_ckpt'], test_time_momentum=args.test_time_momentum)
        cam_list = [args.cam1] # Change this behavior. Pass list in hydra configs
        if args.cam2 is not None:
            cam_list.append(args.cam2)
            if args.cam3 is not None:
                cam_list.append(args.cam3)

        num_cam = len(cam_list)
        print(colored(f"Cameras : {cam_list}", "red"))
        print(colored(f"Use hand state: {job_data['hybrid_state']}", "cyan"))
        camera_type = cam_list[0]
        if num_cam > 1:
            camera_type = "multicam"
        e, env_kwargs = make_env(job_data['env']+'-v0', from_pixels=True, cam_list=cam_list, 
                        encoder=encoder, encoder_type=job_data['encoder_type'], hybrid_state=job_data['hybrid_state'], 
                        episode_length=args.episode_length)
    else :
        e, env_kwargs = make_env(job_data['env'])
        

    policy = MLP(e.spec, hidden_sizes=job_data['policy_size'], seed=job_data['seed'])
    baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=job_data['vf_batch_size'],
                           epochs=job_data['vf_epochs'], learn_rate=job_data['vf_learn_rate'], use_gpu=True)
    
    # Get demonstration data if necessary and behavior clone
    if job_data['algorithm'] != 'NPG':
        use_BC = args.use_bc
        use_encoder_adaptation = args.use_encoder_adaptation
        print(colored(f"===========Use BC: {use_BC}===========", "red"))

        if use_BC or use_encoder_adaptation:
            print("========================================")
            print("Creating expert policy and execution environment")
            print("========================================")

            available_cameras = ['frontview', 'backview', 'sideview']     
            cam_names = [args.cam1,]
            if args.cam2 is not None:
                cam_names.append(args.cam2)
            if args.cam3 is not None:
                cam_names.append(args.cam3)
            
            env_name, object_name = job_data['env'].split("-") if "-" in job_data['env'] else (job_data['env'], None)
            # create environment
            friction = (1, 0.5, 0.01)
            T = args.episode_length
            cprint("episode length: {}".format(T), "red")
            if env_name == "relocate":
                if object_name is None:
                    raise ValueError("For relocate task, object name is needed.")
                # e = YCBRelocate(has_renderer=True, object_name=object_name, friction=friction, object_scale=0.8,
                #                 solref="-6000 -300", randomness_scale=0.25)
                demo_env = YCBRelocate(has_renderer=False, object_name=object_name, friction=friction, object_scale=0.8,
                                solref="-6000 -300", randomness_scale=0.25)
            elif env_name == "pour":
                demo_env = WaterPouringEnv(has_renderer=False, scale=1.0, tank_size=(0.15, 0.15, 0.08))
            elif env_name == "place_inside":
                demo_env = MugPlaceObjectEnv(has_renderer=False, object_scale=0.8, mug_scale=1.5)
            else:
                raise NotImplementedError

            # create policy
            if env_name == "relocate":
                expert_policy_path = f"../../third_party/dexmv-sim/pretrained_model/{env_name}-{object_name}.pickle"
            else:
                expert_policy_path = f"../../third_party/dexmv-sim/pretrained_model/{env_name}.pickle"
            print("expert policy path : ", expert_policy_path)
            expert_policy = pickle.load(open(expert_policy_path, 'rb'))


            print("========================================")
            print("Collecting expert demonstrations")
            print("========================================")
            
            mode = "exploration"
            num_demos = args.num_demos
            print("Number of demos : ", num_demos)
            
            # define a function to extract features from images
            encoder_preprocess = encoder.get_transform()
            def get_feature(obs_img:np.ndarray):
                """
                feature for single step
                """
                obs_img = torch.from_numpy(obs_img).float().to('cuda')
                obs_img = obs_img.div(255.0).permute(0, 3, 1, 2)
                obs_img = encoder_preprocess(obs_img)
                feature = encoder.get_features(obs_img)
                return feature.reshape(-1)

            # start collecting demos
            new_demo_paths = []
            img_video_list = []
            demo_env.set_seed(job_data['seed'])
            start_time = timer.time()
            for demo_id in tqdm(range(num_demos), desc="Collecting demos"):
                obs, done = demo_env.reset(), False
                obs_img = []
                for cam in cam_names:
                    obs_img.append(render(env=demo_env, img_size=256, cam=cam))
                obs_img = np.stack(obs_img, axis=0)
                obs_feature = get_feature(obs_img)
                state_feature = demo_env.data.qpos.ravel()[:30]


                idx = 0
                new_path = {}
                obs_list = [obs]
                obs_img_list = [obs_img]
                obs_feat_list = [obs_feature]
                obs_state_list = [state_feature] # robo qpos
                reward_list = []
                action_list = []

                ep_reward = 0
                
                while not done:
                    action = expert_policy.get_action(obs)[0] if mode == 'exploration' else expert_policy.get_action(obs)[1]['evaluation']
                    next_obs, reward, done, info = demo_env.step(action)
                    ep_reward += reward
                    obs_list.append(next_obs)
                    action_list.append(action)
                    reward_list.append(reward)
                    
                    # generate visual feature
                    obs_img = []
                    for cam in cam_names:
                        obs_img.append(render(env=demo_env, img_size=256, cam=cam))
                    obs_img = np.stack(obs_img, axis=0)
                    obs_feature = get_feature(obs_img)
                    obs_feat_list.append(obs_feature)

                    if use_encoder_adaptation:
                        obs_img_list.append(obs_img) # np array, uint8

                    # generate robo state feature
                    state_feature = demo_env.data.qpos.ravel()[:30]
                    obs_state_list.append(state_feature)

                    obs = next_obs
                    idx += 1
                    if idx >= T or done:
                        break
                
                single_demo = {}
                if args.hybrid_state:
                    visual_feature = np.stack(obs_feat_list, axis=0)
                    robo_state_feature = np.stack(obs_state_list, axis=0)
                    single_demo['observations'] = np.concatenate([visual_feature, robo_state_feature], axis=1)
                else:
                    single_demo['observations'] = np.stack(obs_feat_list, axis=0)
                single_demo['observations'] = single_demo['observations'][:-1] # remove last obs
                single_demo['actions'] = np.stack(action_list, axis=0)
                single_demo['rewards'] = np.stack(reward_list, axis=0)

                new_demo_paths.append(single_demo)

                if use_encoder_adaptation:
                    img_video_list.append(np.stack(obs_img_list, axis=0))

                
            demo_paths = new_demo_paths
            if use_encoder_adaptation:
                img_video_list = np.stack(img_video_list, axis=0)
            print("time taken = %f" % (timer.time() - start_time))
            print("========================================")


            if use_encoder_adaptation:
                # train encoder
                print("========================================")
                print("Adapt encoder")
                print("========================================")
                ts = timer.time()
                test_time_adaptation(encoder=encoder, cfg=args,
                                    videos=img_video_list, 
                                    lr=args.adaptation_lr, num_iter=args.adaptation_num_iter,
                                    batch_size=args.adaptation_batch_size, 
                                    init_model_path=args.adaptation_init_model_path,
                                    freeze_conv=args.adaptation_freeze_conv,)
                if not encoder.adapted:
                    cprint("Encoder adaptation failed", 'red')
                print("========================================")
                print("Encoder adaptatation complete !!!")
                print("time taken = %f" % (timer.time() - ts))
                print("========================================")

            if use_BC:
                bc_agent = BC(demo_paths, policy=policy, encoder=encoder, epochs=job_data['bc_epochs'], batch_size=job_data['bc_batch_size'],
                            lr=job_data['bc_learn_rate'], loss_type='MSE', set_transforms=False)
                in_shift, in_scale, out_shift, out_scale = bc_agent.compute_transformations()
                bc_agent.set_transformations(in_shift, in_scale, out_shift, out_scale)
                bc_agent.set_variance_with_data(out_scale)
                ts = timer.time()
                print("Running BC with expert demonstrations")
                print("========================================")
                bc_agent.train()
                print("========================================")
                print("BC training complete !!!")
                print("time taken = %f" % (timer.time() - ts))
                print("========================================")




            if job_data['eval_rollouts'] >= 1:
                score = e.evaluate_policy(policy, num_episodes=job_data['eval_rollouts'], mean_action=True)
                print("Score with behavior cloning = %f" % score['return_mean'])
                print("Success Rate with behavior cloning = %f" % score['success_rate'])
                pickle.dump(policy, open(JOB_DIR + '/policy_bs{}_epochs{}.pickle'.format(job_data['bc_batch_size'], job_data['bc_epochs']), 'wb'))
        else:
            demo_paths = None
    
    if job_data['algorithm'] != 'DAPG':
        # We throw away the demo data when training from scratch or fine-tuning with RL without explicit augmentation
        demo_paths = None

    # ===============================================================================
    # RL Loop
    # ===============================================================================
    # policy.log_std_val *= 3.
    rl_agent = DAPG(e, policy, baseline, demo_paths,
                    normalized_step_size=job_data['rl_step_size'],
                    lam_0=job_data['lam_0'], lam_1=job_data['lam_1'],
                    seed=job_data['seed'], save_logs=True, use_wandb=args.use_wandb
                    )

    print("========================================")
    print("Starting reinforcement learning phase")
    print("========================================")

    ts = timer.time()
    train_agent(job_name=JOB_DIR,
                agent=rl_agent,
                seed=job_data['seed'],
                niter=job_data['rl_num_iter'],
                gamma=job_data['rl_gamma'],
                gae_lambda=job_data['rl_gae'],
                num_cpu=job_data['num_cpu'],
                sample_mode='trajectories',
                num_traj=job_data['rl_num_traj'],
                save_freq=job_data['save_freq'],
                evaluation_rollouts=job_data['eval_rollouts'],
                env_kwargs=env_kwargs,
                use_wandb=args.use_wandb)
    print("time taken = %f" % (timer.time()-ts))


if __name__ == '__main__' :
    multiprocessing.set_start_method('spawn')
    main()
