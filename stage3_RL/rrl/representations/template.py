import torch
import torch.nn as nn
import torchvision.transforms as transforms

from termcolor import colored

class TemplateEncoder(nn.Module):

    def __init__(self):
        super(TemplateEncoder, self).__init__()
        self.encoder = lambda x: x
        self.latent_dim = None
        pass

    def forward(self, x):
        x = self.encoder(x)
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
