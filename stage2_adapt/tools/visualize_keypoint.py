import sys

def info(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
    # we are in interactive mode or we don't have a tty-like
    # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, pdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        print
        # ...then start the debugger in post-mortem mode.
        # pdb.pm() # deprecated
        pdb.post_mortem(tb) # more "modern"

sys.excepthook = info

import argparse
import os
import pprint
import shutil
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision

import _init_paths
from config import cfg
from config import update_config
from core.loss import FinetuneUnsupLoss
from core.function import test_time_training
from utils.utils import get_optimizer

import dataset
import models
from core.inference import get_final_preds, get_max_preds
import wandb
from termcolor import colored, cprint
from natsort import natsorted
import matplotlib.pyplot as plt
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    parser.add_argument('--keypointnet_pretrain',
                        type=str,
                        default='none')

    parser.add_argument('--freeze_bn',
                        default=0,
                        type=int)

    
    parser.add_argument('--use_entire_pretrain', default=0, type=int)
    parser.add_argument('--freeze_encoder', default=0, type=int)
    parser.add_argument('--resume', default=0, type=int)
    
    # wandb
    parser.add_argument('--use_wandb', default=0, type=int)
    parser.add_argument('--wandb_project', default='handae', type=str)
    parser.add_argument('--wandb_group', default='debug', type=str)
    parser.add_argument('--wandb_name', default='0', type=str)
    parser.add_argument('--task_name', type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)


    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    

    final_output_dir = os.path.join("logs", args.wandb_group)
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)
    print(colored('final output dir: {}'.format(final_output_dir), 'cyan'))

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True, is_finetune=False, freeze_bn=False, freeze_encoder=args.freeze_encoder
    )

    if args.use_entire_pretrain:
        # Freeze layer_to_freeze and bn
        layer_to_freeze = ['final_layer', 'sup_f', 'sup_query', 'mh_attn', 'sup_weight']
        for layer in layer_to_freeze:
            if not hasattr(model.pose_net, layer):
                continue
            for p in getattr(model.pose_net, layer).parameters():
                p.requires_grad = False

        if args.freeze_bn:
            # Let's try to freeze more
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                    for p in m.parameters():
                        p.requires_grad = False
        

        try:
            model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
        except:
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
        cprint('=> loaded entire model {}'.format(cfg.TEST.MODEL_FILE), 'cyan')

    else:
        cprint('=> not loading entire model', 'cyan')
    # if len(cfg.GPUS) > 1:
    #     raise NotImplementedError
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    
    if args.keypointnet_pretrain != 'none':
        assert os.path.exists(args.keypointnet_pretrain), "keypointnet_pretrain not found"
        keypointnet_ckpt = torch.load(args.keypointnet_pretrain, map_location='cpu')
        state_dict = {}
        for k, v in keypointnet_ckpt.items():
            if 'main_encoder' in k:
                k = k.replace('main_encoder.', '')
                state_dict[k] = v
        model.module.pose_net.load_state_dict(state_dict, strict=False)
        cprint(f'=> loaded keypoint-net pretrain from {args.keypointnet_pretrain}', 'cyan')
        
        # debug
        # # print whether the model's param needs grad
        # for name, param in model.module.pose_net.named_parameters():
        #     print(name, param.requires_grad)


    if args.resume:
        # Load model
        resume_ckpt = os.path.join(final_output_dir, 'ckpt', 'latest.pth')
        model_state_dict = torch.load(resume_ckpt)
        model.load_state_dict(model_state_dict)
        print(colored('=> resume model from {}'.format(resume_ckpt), 'red'))

    if not cfg.MODEL.IS_IMM:
        assert cfg.TEST.TTP_WITH_SUP, "Pose only for TTP only works with supervised frame"

    # define loss function (criterion) and optimizer
    criterion = FinetuneUnsupLoss(cfg).cuda()

    # Data loading code
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # )
    # 224 img size to align RL part
    resized_center_crop = transforms.Compose([
        transforms.Resize(224),
        # transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    
    resize_448 = transforms.Compose([
        transforms.Resize(224),
    ])
    model = model.module
    model = model.module
    model.eval()

    root_path = os.getcwd().split('/')[:-1]
    root_path = '/'.join(root_path)



    video_idx = 0
    video_path = f"{root_path}/AdroitImgDataset/{args.task_name}/{video_idx}"
    # video_path = "/home/yanjieze/projects/HandAutoencoder/human_files/imgs"



    # load all images
    images = []
    save_dir = "visualization_buffer"
    os.makedirs(save_dir, exist_ok=True)
    # remove all old images
    for img_name in os.listdir(save_dir):
        os.remove(os.path.join(save_dir, img_name))
    
        
    idx = 0
    last_pred_unsup = None

    color = (240/255, 61/255, 62/255)
    marker = '*'
    
    for img_name in tqdm(natsorted(os.listdir(video_path))):
        img_path = os.path.join(video_path, img_name)
        # img_path = "/home/yanjieze/projects/HandAutoencoder/human_hand1.png"
        img = torchvision.io.read_image(img_path)[:3, :, :]
        img = img.cuda().div(255)
        img = resized_center_crop(img)
        images.append(img)

        pose_unsup = model.predict_keypoint(img.unsqueeze(0))
        img = resize_448(img)
        pred_unsup, _ = get_max_preds(pose_unsup.detach().cpu().numpy()) # on a 64x64 map

        pred_unsup = pred_unsup * 4
        # filter out the points that are out of interest region
        # x: 25-224, y: 0-200
        pred_unsup = pred_unsup[:, (pred_unsup[0, :, 0] > 25 ) & (pred_unsup[0, :, 0] < 224 ) & (pred_unsup[0, :, 1] > 0) & (pred_unsup[0, :, 1] < 200 )]

        VIS_KEYPOINT_MAP = False
        if VIS_KEYPOINT_MAP:
            # create a ##38608a image and scatter the keypoints
            keypoint_map = torch.zeros((64, 64, 3))
            keypoint_map[:, :, 0] = 56/255
            keypoint_map[:, :, 1] = 96/255
            keypoint_map[:, :, 2] = 138/255

            plt.figure(figsize=(10, 10))
            plt.imshow(keypoint_map.cpu().numpy(), cmap='gray')
            plt.scatter(pred_unsup[0, :, 0]/4, pred_unsup[0, :, 1]/4, s=600, c=color, marker=',')
            plt.axis('off')
            plt.savefig('keypoint_map.png', bbox_inches='tight', pad_inches=0)
            import ipdb; ipdb.set_trace()
            
        
        
        # pred_unsup = pred_unsup * 8

        

        # visualize
        plt.figure(figsize=(10, 10))
        plt.imshow(img.permute(1, 2, 0).cpu().numpy())


        if idx == 0:
            plt.scatter(pred_unsup[0, :, 0], pred_unsup[0, :, 1], s=500, c=color, marker=marker)
        else: # visualize tracking
            plt.scatter(pred_unsup[0, :, 0], pred_unsup[0, :, 1], s=500, c=color, marker=marker)
            # square
            # plt.scatter(last_pred_unsup[0, :, 0], last_pred_unsup[0, :, 1], s=50, c='blue', marker='x')
            # connect
            # for i in range(last_pred_unsup.shape[1]):
            #     plt.plot([pred_unsup[0, i, 0], last_pred_unsup[0, i, 0]], [pred_unsup[0, i, 1], last_pred_unsup[0, i, 1]], c='b')
        last_pred_unsup = pred_unsup
        plt.tight_layout()
        # remove x and y ticks
        plt.xticks([])
        plt.yticks([])
        # remove all black border
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        # remove blank space
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.savefig(os.path.join(save_dir, img_name))
        plt.close()
        idx += 1
        # if idx == 100:
            # break

    
    # read all imgs and save as mp4. using torchvision
    
    # read all imgs with torchvision
    img_list = []
    for img_name in tqdm(natsorted(os.listdir(save_dir))):
        img_path = os.path.join(save_dir, img_name)
        img = torchvision.io.read_image(img_path)
        img = img[:3]
        img = img.permute(1, 2, 0)
        img_list.append(img)
    

    # save as mp4
    saving_dir = "visualizations_processed"
    os.makedirs(saving_dir, exist_ok=True)

    torchvision.io.write_video(f"{saving_dir}/keypoint_{args.task_name}.mp4", torch.stack(img_list), 20)
    cprint(f'=> done. saved to vis_video_{args.task_name}.mp4', 'cyan')




if __name__ == '__main__':
    main()
