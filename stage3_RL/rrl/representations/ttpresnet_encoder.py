import torch
import torch.nn as nn
import torchvision.transforms as transforms
import omegaconf

from .TTP.imm_joint_model_resnet import get_pose_net

from termcolor import colored

class TTPResnetEncoder(nn.Module):

    def __init__(self, cfg_path, ckpt_path):
        super(TTPResnetEncoder, self).__init__()
        cfg = omegaconf.OmegaConf.load(cfg_path)
        self.pose_net = get_pose_net(cfg, is_train=False, is_finetune=False, freeze_bn=False)
        self.pose_net = nn.DataParallel(self.pose_net)
        self.pose_net.load_state_dict(torch.load(ckpt_path), strict=True)

        self.encoder = self.pose_net.module.image_encoder

        print(colored("[TTPEncoder] Encoder loaded from {}".format(ckpt_path), "cyan"))

        self.latent_dim = 512
        self.cuda()

    def forward(self, x):
        x = self.encoder(x) # 512x14x14
        # downsample to 512
        x = nn.AvgPool2d(14)(x).view(-1, 512)
        return x

    def get_transform(self):
        # need to align pretraining progress !!!
        trans = nn.Sequential(transforms.Resize(256),
                                transforms.CenterCrop(224))
        return trans
    
    def get_features(self, x):
        with torch.no_grad():
            z = self.forward(x)
        return z.cpu().data.numpy() ### Can't store everything in GPU :/
