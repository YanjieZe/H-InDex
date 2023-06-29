import torch
import torch.nn as nn
import torchvision.transforms as transforms
import mvp



from termcolor import colored


class MVPEncoder(nn.Module):

    def __init__(self, ckpt_path):
        super(MVPEncoder, self).__init__()
        model = mvp.load("vits-mae-hoi", ckpt_path)
        model.freeze()
        print(colored(f"[MVPEncoder] Loaded MVP encoder from {ckpt_path}", "green"))

        self.encoder = model
        self.latent_dim = 384

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
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
