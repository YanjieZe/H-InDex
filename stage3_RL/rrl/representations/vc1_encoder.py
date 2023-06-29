import torch
import torch.nn as nn
import torchvision.transforms as transforms

from termcolor import colored


class VC1Encoder(nn.Module):

    def __init__(self, ckpt_path=None):
        super(VC1Encoder, self).__init__()
        from vc_models.models.vit import model_utils
        # model,embd_size,model_transforms,model_info = model_utils.load_model(model_utils.VC1_LARGE_NAME)
        model,embd_size,model_transforms,model_info = model_utils.load_model(model_utils.VC1_BASE_NAME)
        model.eval()

        print(colored(f"[VC1Encoder] Loaded VC1 encoder ViT/B.", "green"))

        self.encoder = model
        self.latent_dim = embd_size

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
