import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from rrl.adaptation_models.imm_joint_model import get_pose_net
from rrl.adaptation_models.perceptual_loss import FinetuneUnsupLoss
from rrl.adaptation_models.inference import get_final_preds, get_max_preds
from termcolor import cprint
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def PSNR(img1, img2, max_val=1):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = max_val
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    
def sample_batch(videos, batch_size, interval=4):
    """
    sample a batch of videos
    """
    # sample a batch of videos
    batch_videos = []
    for _ in range(batch_size):
        idx = np.random.randint(0, len(videos))
        selected_video = videos[idx]

        # squeeze the cam_num channel
        selected_video = selected_video[:, 0]

        # random select src and tgt frames
        # the interval between src and tgt frames should be larger than 4
        src_idx = np.random.randint(0, selected_video.shape[0] - interval)
        tgt_idx = np.random.randint(src_idx + interval, selected_video.shape[0])

        # select src and tgt frames
        src_frame = selected_video[src_idx]
        tgt_frame = selected_video[tgt_idx]

        # stack src and tgt frames
        batch_videos.append(np.stack([src_frame, tgt_frame], axis=0))

    batch_videos = np.stack(batch_videos, axis=0)
    return batch_videos


def test_time_adaptation(encoder, cfg,
         videos, lr=1e-4, num_iter=1000, batch_size=32, init_model_path=None, freeze_conv=True):
    """
    unsupervised keypoint feature adaptation

    videos: shape (num_videos, timestep, num_frames, 256, 256, 3)
    """

    # create Human pose model
    model = get_pose_net(cfg, is_train=True, is_finetune=False, freeze_bn=False, freeze_encoder=False)


    # load pretrained human pose model
    if init_model_path is not None:
        cprint('=> loading human pose model from {}'.format(init_model_path), 'cyan')
        model.load_state_dict(torch.load(init_model_path), strict=True)
    
    # replace the encoder with our encoder
    replace_layer = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']
    for layer in replace_layer:
        if not hasattr(model.pose_net, layer):
            continue
        setattr(model.pose_net, layer, getattr(encoder.model, layer))
        cprint('=> replace layer {} with our encoder'.format(layer), 'green')
        # print whether the layer is frozen
        for p in getattr(model.pose_net, layer).parameters():
            if not p.requires_grad:
                # unfreeze the layer
                p.requires_grad = True


    # freeze some layers
    layer_to_freeze = ['final_layer', 'sup_f', 'sup_query', 'mh_attn', 'sup_weight']
    for layer in layer_to_freeze:
        if not hasattr(model.pose_net, layer):
            continue
        for p in getattr(model.pose_net, layer).parameters():
            p.requires_grad = False
        
    # freeze conv layers
    if freeze_conv:
        # Freeze all convs in model.module.pose_net
        for m in model.pose_net.modules():
            if isinstance(m, nn.Conv2d):
                for p in m.parameters():
                    p.requires_grad = False
        cprint('=> freeze all convs in model.pose_net', 'green')
    

            
    # to cuda
    model = model.cuda()

    model.eval()

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    cprint("=> create adaptation optimizer with lr {}".format(lr), 'cyan')
    # create loss function

    criterion = FinetuneUnsupLoss(cfg).cuda()

    # create transform function
    # transform = transforms.Compose([
    #     transforms.Resize(256), # resize to 256
    #     transforms.CenterCrop(224), # crop to 224
    # ])
    transform = encoder.get_transform()

    # create logging dir
    use_visualize = False
    if use_visualize:
        output_dir = "keypoint_adaptation_visulization"
        os.makedirs(output_dir, exist_ok=True)

    # imgnet normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()
    
    # runnning
    loss = 0.
    psnr = 0.
    progress_bar = tqdm(range(num_iter), desc="Adaptation")

    for i in progress_bar:
        
        # sample a batch of videos
        input = sample_batch(videos, batch_size) # batch_size x 2 x 256 x 256 x 3, np array

        # convert to tensor
        input = torch.from_numpy(input).float().cuda(non_blocking=True)
        input = input.permute(0, 1, 4, 2, 3) # batch_size x 2 x 3 x 256 x 256

        # [0, 255] -> [0, 1], and normalize
        input = input / 255.0
        input = transform(input.view(-1, 3, 256, 256)).view(-1, 2, 3, 224, 224)
        
        # forward
        pred_images, pose_unsup, pose_sup = model(input)

        # pred_images also need to be imgnet normalized
        pred_images = (pred_images - mean) / std

        # compute loss
        images_ref = input[:, 0, :, :].cuda(non_blocking=True)
        images_tgt = input[:, 1, :, :].cuda(non_blocking=True)
        loss = criterion(images=images_tgt, pred_images=pred_images)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        if i % 10 ==0:
            pred_unsup, _ = get_final_preds(pose_unsup.detach().cpu().numpy(), scale=4.0)

            # imgnet normalize reverse
            pred_images = pred_images * std + mean
            pred_images = pred_images.permute(0, 2, 3, 1).detach().cpu().numpy()

            src_images = input[:, 0, :, :]
            src_images = src_images * std + mean
            src_images = src_images.permute(0, 2, 3, 1).detach().cpu().numpy()

            target_images = input[:, 1, :, :]
            target_images = target_images * std + mean
            target_images = target_images.permute(0, 2, 3, 1).detach().cpu().numpy()
            
            psnr = PSNR(pred_images, target_images) # all in [0, 1]

            
            if use_visualize:
                # visualize
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
                plt.title(f"psnr {psnr:.2f} and keypoint")
                plt.scatter(pred_unsup[0, :, 0], pred_unsup[0, :, 1], c='r', s=10)

                plt.tight_layout()

                # save
                plt.savefig(os.path.join(output_dir, f"{i}.png"))
                cprint(f"save {i}.png with psnr {psnr:.2f}", 'green')

        if i % 10 == 0:
            # update tqdm
            progress_bar.set_description(f"Adaptation | loss {loss.item():.4f} | psnr {psnr:.2f}")


    # copy the trained model to the original encoder
    replace_layer = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']
    for layer in replace_layer:
        if not hasattr(encoder.model, layer):
            continue
        setattr(encoder.model, layer, getattr(model.pose_net, layer))
        cprint('=> copy layer {} to our encoder'.format(layer), 'green')
    
    cprint("=> adaptation finished", "cyan")
    # set the encoder to adapted
    encoder.adapted = True

    return encoder