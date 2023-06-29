from itertools import islice

import time
import logging
import os

import numpy as np
import torch

from core.evaluate import accuracy
from core.inference import get_final_preds, get_max_preds
from core.loss import FinetuneUnsupLoss, JointsMSELoss
from utils.transforms import flip_back
from utils.vis import save_debug_images

import matplotlib.pyplot as plt
import wandb
from termcolor import colored, cprint



logger = logging.getLogger(__name__)

def PSNR(img1, img2, max_val=1):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = max_val
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def test_time_training(config, loader, dataset, model, model_state_dict,
          criterion, optimizer, optimizer_state_dict, output_dir, use_wandb=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    model.eval()

    # for ttp dataset, dataset.length == len(dataset) // ttp_batchsize
    num_samples = dataset.length // config.TEST.DOWNSAMPLE
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    idx = 0

    print("Number of samples: {}".format(num_samples))

    end = time.time()
    num_epochs = config.TRAIN.NUM_EPOCHS
    num_iters = config.TRAIN.NUM_ITERS
    num_iter = 0
    eval_freq = config.TRAIN.EVAL_FREQ
    
    loss_function = torch.nn.MSELoss()

    data_iter = iter(loader)


    # for epoch in range(num_epochs):
    for _ in range(num_iters):
        # for i, (input, target, target_weight, meta) in enumerate(loader):
        try:
            input, target, target_weight, meta = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            input, target, target_weight, meta = next(data_iter)
        

        # measure data loading time
        data_time.update(time.time() - end)
        num_iter += 1

        # Save the input for future testing
        # input_queue.append((i, input, target, target_weight, meta))

        
        # compute output
        pred_images, pose_unsup, pose_sup = model(input)

        images_ref = input[:, 0, :, :].cuda(non_blocking=True)
        images_tgt = input[:, 1, :, :].cuda(non_blocking=True)
        
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)
        # _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
        #                                 target.detach().cpu().numpy())
        pred, _ = get_max_preds(pose_unsup.detach().cpu().numpy())
        loss_percep = criterion(images=images_tgt, pred_images=pred_images)

        loss = loss_percep
        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if num_iter % 100 == 0:
            print(f"Iter: {num_iter}, Peceptual Loss: {loss_percep:.4f}, total loss: {loss:.4f}")

        if num_iter % eval_freq == 0:
            with torch.no_grad():
                pred_images, pose_unsup, pose_sup = model(input)
                pred_unsup, _ = get_max_preds(pose_unsup.detach().cpu().numpy()) # on a 64x64 map
                pred_sup, _ = get_max_preds(pose_sup.detach().cpu().numpy()) # on a 64x64 map
                # to 256x256 map
                pred_unsup = pred_unsup * 4
                pred_sup = pred_sup * 4

                # # imgnet normalize reverse
                # pred_images = pred_images * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda() + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
                pred_images = pred_images.permute(0, 2, 3, 1).cpu().numpy()
                src_images = input[:, 0, :, :]
                # src_images = src_images * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)+ torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                src_images = src_images.permute(0, 2, 3, 1).cpu().numpy()
                target_images = input[:, 1, :, :]
                # target_images = target_images * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                target_images = target_images.permute(0, 2, 3, 1).cpu().numpy()

            # compute PSNR
            psnr = PSNR(pred_images, target_images)
            print(colored("[Eval] PSNR: {}".format(psnr), "green"))

            SAVE_AS_ONE_FIG = False
            if SAVE_AS_ONE_FIG:
                # show src, target, pred + unsup keypoint, pred + sup keypoint in subplot
                plt.figure()
                plt.subplot(1, 3, 1)
                # plt.imshow(src_images[0][..., ::-1])
                plt.imshow(src_images[0])
                plt.title("src")

                plt.subplot(1, 3, 2)
                # plt.imshow(target_images[0][..., ::-1])
                plt.imshow(target_images[0])
                plt.title("target")

                plt.subplot(1, 3, 3)
                # plt.imshow(pred_images[0][..., ::-1])
                plt.imshow(pred_images[0])
                plt.title(f"psnr:{psnr:.2f} w. keypoint")
                plt.scatter(pred_unsup[0, :, 0], pred_unsup[0, :, 1], c='r', s=10)

                plt.tight_layout()
                
                # save the image
                imgdir = os.path.join(output_dir, "pred_images")
                os.makedirs(imgdir, exist_ok=True)
                plt.savefig(os.path.join(imgdir, "{}.png".format(num_iter)))

                print(colored("[Eval] Saved image to pred_images/{}.png".format(num_iter), "green"))
            else:
                # save src, target, pred + unsup keypoint separately
                imgdir = os.path.join(output_dir, "pred_images")
                os.makedirs(imgdir, exist_ok=True)
                # save as step_src.png, step_target.png, step_pred.png
                plt.imsave(os.path.join(imgdir, "{}_src.png".format(num_iter)), src_images[0])
                plt.imsave(os.path.join(imgdir, "{}_target.png".format(num_iter)), target_images[0])
                plt.imsave(os.path.join(imgdir, "{}_pred.png".format(num_iter)), pred_images[0].clip(0, 1))

                cprint("[Eval] Saved image to pred_images/{}_xxx.png".format(num_iter), "green")

            if use_wandb:
                wandb.log({"psnr": psnr})

            

            # # plot attention gt and predin subplot
            # plt.figure()
            # plt.subplot(1, 2, 1)
            # plt.imshow(gt_attention[0].cpu().numpy())
            # plt.title("gt_attention")
            
            # plt.subplot(1, 2, 2)
            # plt.imshow(pred_attentions[0].cpu().numpy())
            # plt.title("pred_attention")
            # # save the image
            # imgdir = os.path.join(output_dir, "pred_images")
            # os.makedirs(imgdir, exist_ok=True)
            # plt.savefig(os.path.join(imgdir, "{}_attention.png".format(num_iter)))

            # print(colored("[Eval] Saved image to pred_images/{}_attention.png".format(num_iter), "green"))
        

        if num_iter % eval_freq == 0:
            # save ckpt
            ckptdir = os.path.join(output_dir, "ckpt")
            os.makedirs(ckptdir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckptdir, "latest.pth"))
            print(colored(f"[Eval] Saved ckpt to {ckptdir}/latest.pth", "green"))

    # final save
    ckptdir = os.path.join(output_dir, "ckpt")
    os.makedirs(ckptdir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckptdir, "latest.pth"))
    print(colored(f"[Eval] Saved ckpt to {ckptdir}/latest.pth", "green"))
    return None




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0