import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import resnet34, resnet50
from PIL import Image
import torch.nn.functional as F
from termcolor import colored
import rrl.utils as utils

_encoders = {'resnet34' : resnet34, 'resnet50' : resnet50}

LATENT_DIM = {'resnet34' : 512, 'resnet50' : 2048}

class HandAutoEncoder(nn.Module):
    def __init__(self):
        super(HandAutoEncoder, self).__init__()

        encoder_type = 'resnet50'
        self.img_size = 224
        self.encoder_type = encoder_type

        if self.encoder_type in _encoders :
            self.encoder = _encoders[encoder_type](pretrained=True)
        else :
            print("Please enter a valid encoder type")
            raise Exception
        
        self.latent_dim = LATENT_DIM[encoder_type] + 30
        for param in self.encoder.parameters():
            param.requires_grad = False
        if self.encoder_type in _encoders :
            num_ftrs = self.encoder.fc.in_features
            self.num_ftrs = num_ftrs
            self.encoder.fc = Identity()

        
        # Decoder
        self.joint_dim = 30
        hidden_dims = [2048+self.joint_dim, 1024, 512, 256, 256, 128, 128, 64]
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1], 
                        kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU())
            )
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1], 3,
                    kernel_size=3, stride=2, padding=1, output_padding=1),
            )
        )
        self.decoder = nn.Sequential(*modules)

        # joint predictor
        self.joint_predictor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, self.joint_dim))

        # free first two layers of encoder
        for param in self.encoder.layer1.parameters():
            param.requires_grad = False
        for param in self.encoder.layer2.parameters():
            param.requires_grad = False

        # loss function
        self.perceptual_loss_function = utils.PerceptualLoss()
        self.recon_loss_function = nn.MSELoss()

        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)



    def get_transform(self):
        trans = nn.Sequential(transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225]))
        return trans
    

    def get_features(self, x):
        with torch.no_grad():
            z = self.forward_encoder(x)
            joint_pred = self.joint_predictor(z)
            z = torch.cat([z, joint_pred], dim=1)
        return z.cpu().data.numpy() ### Can't store everything in GPU :/


    def forward_encoder(self, x):
        return self.encoder(x)
    

    def forward_decoder(self, x):
        decoded = self.decoder(x.view(x.size(0), -1, 1, 1))
        # interpolate to the original size
        decoded = F.interpolate(decoded, size=(self.img_size, self.img_size), mode='bilinear', align_corners=True)
        return decoded


    def forward(self, x):
        latent = self.forward_encoder(x)
        latent_joint = self.joint_predictor(latent)
        latent = torch.cat([latent, latent_joint], dim=1)
        decoded = self.forward_decoder(latent.view(latent.size(0), -1, 1, 1))
        return decoded
    

    def forward_dynamic(self, src, tgt):
        """
        recon tgt from src
        """
        latent_src = self.forward_encoder(src)
        latent_tgt = self.forward_encoder(tgt)
        latent_joint_tgt = self.joint_predictor(latent_tgt)
        latent= torch.cat([latent_src, latent_joint_tgt], dim=1)
        decoded2 = self.forward_decoder(latent.view(latent.size(0), -1, 1, 1))
        return decoded2
    

    def compute_loss(self, batch_img1, batch_img2):
        # encoding
        latent_img1 = self.forward_encoder(batch_img1)
        latent_img2 = self.forward_encoder(batch_img2)

        # joint prediction
        latent_joint1 = self.joint_predictor(latent_img1)
        latent_joint2 = self.joint_predictor(latent_img2)


        # decoding for recon
        recon_img1 = self.forward_decoder(torch.cat([latent_img1, latent_joint1], dim=1))
        recon_img2 = self.forward_decoder(torch.cat([latent_img2, latent_joint2], dim=1))
        # recon loss and perceptual loss
        # perceptual_loss_by_recon = self.perceptual_loss_function(batch_img1, recon_img1) + self.perceptual_loss_function(batch_img2, recon_img2)
        recon_loss_by_recon = self.recon_loss_function(recon_img1, batch_img1) + self.recon_loss_function(recon_img2, batch_img2)
        
        
        # decoding for dynamic
        recon_img2from1 = self.forward_decoder(torch.cat([latent_img1, latent_joint2], dim=1))
        recon_img1from2 = self.forward_decoder(torch.cat([latent_img2, latent_joint1], dim=1))
        # recon loss and perceptual loss
        # perceptual_loss_by_dynamic = self.perceptual_loss_function(batch_img1, recon_img1from2) + self.perceptual_loss_function(batch_img2, recon_img2from1)
        recon_loss_by_dynamic = self.recon_loss_function(recon_img1from2, batch_img1) + self.recon_loss_function(recon_img2from1, batch_img2)


        # loss = perceptual_loss_by_recon + recon_loss_by_recon + perceptual_loss_by_dynamic + recon_loss_by_dynamic
        loss = recon_loss_by_recon + recon_loss_by_dynamic
        return {'loss': loss,
            # 'perceptual_loss_by_recon': perceptual_loss_by_recon, 
            'recon_loss_by_recon': recon_loss_by_recon,
            # 'perceptual_loss_by_dynamic': perceptual_loss_by_dynamic, 
            'recon_loss_by_dynamic': recon_loss_by_dynamic}


    def train_step(self, demos):
        # collect obs into batch
        obs_src = []
        obs_tgt = []
        for demo in demos:
            # random sample two frames
            idx = np.random.choice(demo['observations_img'].shape[0], 2, replace=False)
            obs_src.append(demo['observations_img'][idx[0]])
            obs_tgt.append(demo['observations_img'][idx[1]])
        obs_src = torch.stack(obs_src).squeeze(1)
        obs_tgt = torch.stack(obs_tgt).squeeze(1)

        self.train()

        chunk_size = 16
        for i in range(0, obs_src.shape[0], chunk_size):
            self.optimizer.zero_grad()
            loss_dict = self.compute_loss(obs_src[i:i+chunk_size], obs_tgt[i:i+chunk_size])
            loss_dict['loss'].backward()
            self.optimizer.step()

        self.eval()
        return loss_dict


class IdentityEncoder(nn.Module):
    def __init__(self):
        super(IdentityEncoder, self).__init__()

    def forward(self, x):
        return x

    def get_transform(self):
        return transforms.Compose([
                          transforms.ToTensor(),
                          ])

    def get_features(self, x):
        return x.reshape(-1)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
