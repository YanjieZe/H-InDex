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


import _init_paths
from config import cfg
from config import update_config
from core.loss import FinetuneUnsupLoss
from core.function import test_time_training
from utils.utils import get_optimizer

import dataset
import models

import wandb
from termcolor import colored, cprint

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

    parser.add_argument('--freeze_conv',
                        default=0,
                        type=int)

    parser.add_argument('--task_name',
                        type=str,
                        default='none')
    
    parser.add_argument('--use_entire_pretrain', default=0, type=int)
    parser.add_argument('--freeze_encoder', default=0, type=int)
    parser.add_argument('--resume', default=0, type=int)
    
    # wandb
    parser.add_argument('--use_wandb', default=0, type=int)
    parser.add_argument('--wandb_project', default='handae', type=str)
    parser.add_argument('--wandb_group', default='debug', type=str)
    parser.add_argument('--wandb_name', default='0', type=str)

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
    
    if args.use_wandb:
        wandb.init(project="hand_recon", group=args.wandb_group, name=args.wandb_name)
        wandb.config.update(cfg)

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
            cprint('=> Freeze BN', 'cyan')
        

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

    if args.freeze_conv:
        # Freeze all convs in model.module.pose_net
        for m in model.module.pose_net.modules():
            if isinstance(m, nn.Conv2d):
                for p in m.parameters():
                    p.requires_grad = False
        cprint('=> freeze all convs in model.module.pose_net', 'green')

    # # print which modules are not frozen
    # for name, param in model.named_parameters():
    #     if param.requires_grad and 'pose_net' in name:
    #         cprint(f"=> {name} is not frozen", 'cyan')

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
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            resized_center_crop,
            transforms.ToTensor(),]),
        args.task_name,
    )
    dataset[0] # debug
    # import torchvision
    # torchvision.utils.save_image(dataset[0][0][0], 'test.png')

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=True,
        num_workers=cfg.WORKERS,
        pin_memory=False
    )

    optimizer = get_optimizer(cfg, model)

    # use these to re-init after every video
    model_state_dict = copy.deepcopy(model.state_dict())
    optimizer_state_dict = copy.deepcopy(optimizer.state_dict())

    
    print('test time training')
    test_time_training(cfg, loader, dataset, model, model_state_dict, criterion,
            optimizer, optimizer_state_dict, final_output_dir, use_wandb=args.use_wandb)



if __name__ == '__main__':
    main()
