import torch
import torch.nn as nn
import torchvision.transforms as transforms

from .dino import DINO

from termcolor import colored

class DINOEncoder(nn.Module):

    def __init__(self, ckpt_path):
        super(DINOEncoder, self).__init__()
        self.encoder = DINO(pretrain_path=ckpt_path)
        self.latent_dim = 384


    def forward(self, x):
        _, x = self.encoder(x, return_cls_token=True)
        # x = nn.AvgPool2d(28)(x).view(-1, 384)
        return x

    def get_transform(self):
        # need to align pretraining progress !!!
        trans = nn.Sequential(transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225]))
        return trans
    
    def get_features(self, x):
        with torch.no_grad():
            z = self.forward(x)
        return z.cpu().data.numpy() ### Can't store everything in GPU :/
