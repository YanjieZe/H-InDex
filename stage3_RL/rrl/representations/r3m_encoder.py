import torch
import torch.nn as nn
import torchvision.transforms as transforms
from r3m import load_r3m

from termcolor import colored


class R3MEncoder(nn.Module):

    def __init__(self, model_and_config_folder):
        super(R3MEncoder, self).__init__()
        rep = load_r3m("resnet50", model_and_config_folder=model_and_config_folder) # resnet18, resnet34
        print(colored(f"[R3MEncoder] Loaded R3M encoder from {model_and_config_folder}", "green"))
        rep.eval()
        self.encoder = rep
        self.latent_dim = 2048

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder.module.convnet(x)
        return x

    def get_transform(self):
        # need to align pretraining progress !!!
        trans = nn.Sequential(transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        return trans
    
    def get_features(self, x):
        with torch.no_grad():
            z = self.forward(x)
        return z.cpu().data.numpy() ### Can't store everything in GPU :/
