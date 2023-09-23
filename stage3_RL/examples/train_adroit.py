# Copyright (c) Rutav Shah, Indian Institute of Technlogy Kharagpur
# Copyright (c) Facebook, Inc. and its affiliates

"""
This is a job script for running NPG/DAPG on hand tasks and other gym envs.
Note that DAPG generalizes PG and BC init + PG finetuning.
With appropriate settings of parameters, we can recover the full family.
"""

# import sys

# def info(type, value, tb):
#     if hasattr(sys, 'ps1') or not sys.stderr.isatty():
#     # we are in interactive mode or we don't have a tty-like
#     # device, so we call the default hook
#         sys.__excepthook__(type, value, tb)
#     else:
#         import traceback, pdb
#         # we are NOT in interactive mode, print the exception...
#         traceback.print_exception(type, value, tb)
#         print
#         # ...then start the debugger in post-mortem mode.
#         # pdb.pm() # deprecated
#         pdb.post_mortem(tb) # more "modern"

# sys.excepthook = info

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
_mj_envs = {'pen-v0', 'hammer-v0', 'door-v0', 'relocate-v0', }
_dexmv_envs = {'pour-v0', 'place_inside-v0'}
# ===============================================================================
# Get command line arguments
# ===============================================================================

@hydra.main(config_name="hammer_dapg", config_path="config/adroit")
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
        e, env_kwargs = make_env(job_data['env'], from_pixels=True, cam_list=cam_list, 
                        encoder=encoder, encoder_type=job_data['encoder_type'], hybrid_state=job_data['hybrid_state'])
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
            print("Collecting expert demonstrations")
            print("========================================")
            if job_data['env'] in ['pour-v0', 'place-v0']: # dexmv demo
                demo_paths = list(pickle.load(open(job_data['demo_file'], 'rb')).values())
            else: # dapg demo
                demo_paths = pickle.load(open(job_data['demo_file'], 'rb'))
            num_demo = args.num_demos
            demo_paths = demo_paths[:num_demo] # we use full demos

            print("Number of demo paths : ", len(demo_paths))

            # generate feature observation online
            print("========================================")
            print("Generating feature observation for demonstrations")
            new_demo_paths = []
            try :
                e.set_seed(demo_paths[0]['seed'])
            except :
                print("++++++++++++++++++++++++++++ Couldn't find the seed of the demos. Please verify.")
                pass
            start_time = timer.time()
            img_video_list = []
            for path in tqdm(demo_paths, desc="Converting demonstrations"):
                obs,img_obs = e.reset()
                if torch.is_tensor(obs):
                    obs = obs.data.cpu().numpy()
                if job_data['env'] in _mj_envs :
                    e.set_env_state(path["init_state_dict"])
                elif job_data['env'] in _dexmv_envs :
                    e.set_env_state(path["sim_data"][0])
                else :
                    print("Please enter valid environment.")

                idx = 0
                new_path = {}
                obs_list = [obs]
                img_obs_list = [img_obs]
                vis_obs = []
                ep_reward = 0
                for action in path["actions"] :
                    next_obs, next_img_obs, reward, done, info = e.step(action)
                    if torch.is_tensor(next_obs):
                        next_obs = next_obs.data.cpu().numpy()
                    ep_reward += reward
                    obs_list.append(next_obs)
                    if args.use_encoder_adaptation:
                        img_obs_list.append(next_img_obs)
                    # vis_obs.append(next_img_obs)
                    obs = next_obs
                    idx += 1

                # save as video
                # import torchvision
                # video = torch.stack(vis_obs).cpu() # t, c, h, w
                # # reverse imgnet normalization
                # video = video * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                # torchvision.io.write_video("debug_cam1.mp4", video[:,0].permute(0, 2, 3, 1)*255, 30)
                # torchvision.io.write_video("debug1.mp4", video[:,1].permute(0, 2, 3, 1)*255, 30)
                # torchvision.io.write_video("debug2.mp4", video[:,2].permute(0, 2, 3, 1)*255, 30)
                # torchvision.io.write_video("demo1.mp4", video[:,1].permute(0, 2, 3, 1)*255, 30)
                # torchvision.io.write_video("demo2.mp4", video[:,2].permute(0, 2, 3, 1)*255, 30)
                


                new_path_obs = np.stack(obs_list[:-1])
                

                assert new_path_obs.shape[0] == path["observations"].shape[0]
                new_path['observations'] =  new_path_obs
                new_path['actions'] =  path['actions']
                new_path['rewards'] =  path['rewards']
                # new_path['init_state_dict'] =  path['init_state_dict']
                new_demo_paths.append(new_path)

                if args.use_encoder_adaptation:
                    img_obs_list = np.stack(img_obs_list[:-1]).transpose(0, 1, 3, 4, 2)
                    img_video_list.append(img_obs_list)
                assert type(path["observations"]) == type(new_path["observations"])
                
            demo_paths = new_demo_paths
            if args.use_encoder_adaptation:
                # channel has been last
                img_video_list = img_video_list

            print("time taken = %f" % (timer.time() - start_time))
            print("========================================")
            

            # if job_data['encoder_type'] == "hand_autoencoder": # finetune visual encoder
            #     print("========================================")
            #     print("Finetuning visual encoder")
            #     print("========================================")
            #     for ft_step in range(1000):
            #         loss_dict = encoder.train_step(demo_paths)
            #         if ft_step % 100 == 0:
            #             print(colored('visual finetune loss: ', 'red'), loss_dict['loss'].item())

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
                print("========================================")
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
