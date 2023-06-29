import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50

from termcolor import colored

class MoCoEncoder(nn.Module):
    """
    mocov2 (best model)
    """


    def __init__(self, ckpt_path):
        super(MoCoEncoder, self).__init__()
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model = resnet50(pretrained=False)
        # rename moco pre-trained keys
        state_dict = checkpoint["state_dict"]
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith("module.encoder_q") and not k.startswith(
                "module.encoder_q.fc"
            ):
                # remove prefix
                state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        print("=> loaded pre-trained model '{}'".format(ckpt_path))

        # remove the fc layer and classifier layer
        model.fc = nn.Identity()
        model = nn.Sequential(*list(model.children())[:-1])
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
