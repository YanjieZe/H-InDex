import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50

from termcolor import colored

class AlphaPoseEncoder(nn.Module):
    """
    mocov2 (best model)
    """


    def __init__(self, ckpt_path):
        super(AlphaPoseEncoder, self).__init__()
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model = resnet50(pretrained=False)
        # reload pre-trained keys
        new_state = {}
        for k, v in checkpoint.items():
            if k.startswith('preact'):
                new_k = k.replace('preact.', '')
                new_state[new_k] = v
        msg = model.load_state_dict(new_state, strict=False)
        print(msg)
        print(colored("[AlphaPoseEncoder] Loaded pretrained weights from {}".format(ckpt_path), "green"))
        model.fc = nn.Identity()
        self.encoder = model
        self.latent_dim = 2048

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
