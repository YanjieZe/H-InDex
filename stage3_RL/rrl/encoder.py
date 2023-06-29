# Copyright (c) Rutav Shah, Indian Institute of Technlogy Kharagpur
# Copyright (c) Facebook, Inc. and its affiliates

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import resnet34, resnet50
from PIL import Image
from termcolor import colored, cprint
import copy
import rrl.modules as rrl_modules

_encoders = {'resnet34' : resnet34, 'resnet50' : resnet50}

LATENT_DIM = {'resnet34' : 512, 'resnet50' : 2048}


def visualize_feature_map_CHW(feature_map, save_path):
    import matplotlib.pyplot as plt
    feature_map = feature_map.mean(0)
    feature_map = feature_map.cpu().numpy()
    plt.imshow(feature_map)
    plt.savefig(save_path)


class Encoder(nn.Module):
    def __init__(self, encoder_type):
        super(Encoder, self).__init__()
        self.encoder_type = encoder_type
        if self.encoder_type in _encoders :
            self.model = _encoders[encoder_type](pretrained=True)
        else :
            print("Please enter a valid encoder type")
            raise Exception
        
        self.latent_dim = LATENT_DIM[encoder_type]
        for param in self.model.parameters():
            param.requires_grad = False
        if self.encoder_type in _encoders :
            num_ftrs = self.model.fc.in_features
            self.num_ftrs = num_ftrs
            self.model.fc = Identity()

    def forward(self, x):
        x = self.model(x)
        return x

    def get_transform(self):
        trans = nn.Sequential(transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225]))
        return trans
    
    def get_features(self, x):
        with torch.no_grad():
            z = self.model(x)
        return z.cpu().data.numpy() ### Can't store everything in GPU :/

class EncoderBN(nn.Module):
    def __init__(self, encoder_type, bn_momentum=0.01):
        super(EncoderBN, self).__init__()
        self.encoder_type = encoder_type
        if self.encoder_type in _encoders :
            self.model = _encoders[encoder_type](pretrained=True)
        else :
            print("Please enter a valid encoder type")
            raise Exception
        
        self.latent_dim = LATENT_DIM[encoder_type]
        for param in self.model.parameters():
            param.requires_grad = False
        if self.encoder_type in _encoders :
            num_ftrs = self.model.fc.in_features
            self.num_ftrs = num_ftrs
            self.model.fc = Identity()
        
        # use test time batch norm
        use_test_time_bn = True if bn_momentum > 0 else False
        momentum = bn_momentum
        new_model = copy.deepcopy(self.model)
        if use_test_time_bn:
            for module_name, m in self.model.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    # replace original batch norm with test time batch norm
                    new_m = rrl_modules.TestTimeBatchNorm2D(m.num_features, m.eps, momentum, m.affine, m.track_running_stats)
                    # copy parameters
                    new_m.running_mean = m.running_mean
                    new_m.running_var = m.running_var
                    new_m.weight = m.weight
                    new_m.bias = m.bias
                    # replace
                    new_model._modules[module_name] = new_m
            self.model = new_model
            cprint("Use test time batch norm with momentum {}".format(momentum), "yellow")
            
    def forward(self, x):
        x = self.model(x)
        return x

    def get_transform(self):
        trans = nn.Sequential(transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225]))
        return trans
    
    def get_features(self, x):
        with torch.no_grad():
            z = self.model(x)
        return z.cpu().data.numpy() ### Can't store everything in GPU :/

class FrankMocapHandEncoder(nn.Module):
    def __init__(self, encoder_type, ckpt_path, bn_momentum=0.01):
        super(FrankMocapHandEncoder, self).__init__()
        self.encoder_type = encoder_type
        if self.encoder_type in _encoders :
            self.model = _encoders[encoder_type](pretrained=True)
        else :
            print("Please enter a valid encoder type")
            raise Exception
        
        self.latent_dim = LATENT_DIM[encoder_type]
        for param in self.model.parameters():
            param.requires_grad = False
        if self.encoder_type in _encoders :
            num_ftrs = self.model.fc.in_features
            self.num_ftrs = num_ftrs
            self.model.fc = Identity()
        
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = {}
        for k, v in ckpt.items():
            if 'main_encoder' in k:
                k = k.replace('main_encoder.', '')
                state_dict[k] = v
        msg = self.model.load_state_dict(state_dict, strict=False)
        print(msg)
        print(colored("[FrankMocapHandEncoder] Loaded pretrained weights from {}".format(ckpt_path), 'cyan'))

        # remove avg pool
        pooling_method = "avg" # or "max"
        if pooling_method == "max": # experiment results are not good
            self.model.avgpool = nn.AdaptiveMaxPool2d(1)
        elif pooling_method == "avg":
            self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        else:
            raise NotImplementedError
        print(colored("[FrankMocapHandEncoder] Using {} pooling".format(pooling_method), 'cyan'))
        

        # use test time batch norm
        use_test_time_bn = False
        momentum = bn_momentum
        new_model = copy.deepcopy(self.model)
        if use_test_time_bn:
            for module_name, m in self.model.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    # replace original batch norm with test time batch norm
                    new_m = rrl_modules.TestTimeBatchNorm2D(m.num_features, m.eps, momentum, m.affine, m.track_running_stats)
                    # copy parameters
                    new_m.running_mean = m.running_mean
                    new_m.running_var = m.running_var
                    new_m.weight = m.weight
                    new_m.bias = m.bias
                    # replace
                    new_model._modules[module_name] = new_m
            self.model = new_model
                    
        print(colored(f"[FrankMocapHandEncoder] Using test time batch norm: {use_test_time_bn} with momentum: {momentum}", 'cyan'))

        # eval
        self.model.eval()

    def forward(self, x):
        x = self.model(x)
        return x

    def get_transform(self):
        trans = nn.Sequential(transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225]))
        return trans
    
    def get_features(self, x):
        with torch.no_grad():
            z = self.forward(x)
        return z.cpu().data.numpy() ### Can't store everything in GPU :/
    

class FrankMocapHandEncoderBN(nn.Module):
    def __init__(self, encoder_type, ckpt_path, bn_momentum=0.01):
        super(FrankMocapHandEncoderBN, self).__init__()
        self.encoder_type = encoder_type
        if self.encoder_type in _encoders :
            self.model = _encoders[encoder_type](pretrained=True)
        else :
            print("Please enter a valid encoder type")
            raise Exception
        
        self.latent_dim = LATENT_DIM[encoder_type]
        for param in self.model.parameters():
            param.requires_grad = False
        if self.encoder_type in _encoders :
            num_ftrs = self.model.fc.in_features
            self.num_ftrs = num_ftrs
            self.model.fc = Identity()
        
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = {}
        for k, v in ckpt.items():
            if 'main_encoder' in k:
                k = k.replace('main_encoder.', '')
                state_dict[k] = v
        msg = self.model.load_state_dict(state_dict, strict=False)
        print(msg)
        print(colored("[FrankMocapHandEncoder] Loaded pretrained weights from {}".format(ckpt_path), 'cyan'))

        # remove avg pool
        pooling_method = "avg" # or "max"
        if pooling_method == "max": # experiment results are not good
            self.model.avgpool = nn.AdaptiveMaxPool2d(1)
        elif pooling_method == "avg":
            self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        else:
            raise NotImplementedError
        print(colored("[FrankMocapHandEncoder] Using {} pooling".format(pooling_method), 'cyan'))
        

        # use test time batch norm
        use_test_time_bn = True if bn_momentum > 0 else False
        momentum = bn_momentum
        new_model = copy.deepcopy(self.model)
        if use_test_time_bn:
            for module_name, m in self.model.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    # replace original batch norm with test time batch norm
                    new_m = rrl_modules.TestTimeBatchNorm2D(m.num_features, m.eps, momentum, m.affine, m.track_running_stats)
                    # copy parameters
                    new_m.running_mean = m.running_mean
                    new_m.running_var = m.running_var
                    new_m.weight = m.weight
                    new_m.bias = m.bias
                    # replace
                    new_model._modules[module_name] = new_m
            self.model = new_model
      
        print(colored(f"[FrankMocapHandEncoder] Using test time batch norm: {use_test_time_bn} with momentum: {momentum}", 'cyan'))

        # eval
        self.model.eval()

    def forward(self, x):
        x = self.model(x)
        return x

    def get_transform(self):
        trans = nn.Sequential(transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225]))
        return trans
    
    def get_features(self, x):
        with torch.no_grad():
            z = self.forward(x)
        return z.cpu().data.numpy() ### Can't store everything in GPU :/
    


class FrankMocapHandFusionEncoder(nn.Module):
    def __init__(self, encoder_type, ckpt_path):
        super(FrankMocapHandFusionEncoder, self).__init__()
        self.encoder_type = encoder_type
        if self.encoder_type in _encoders :
            self.model = _encoders[encoder_type](pretrained=True)
        else :
            print("Please enter a valid encoder type")
            raise Exception
        
        self.latent_dim = 2048 + 784
        for param in self.model.parameters():
            param.requires_grad = False
        if self.encoder_type in _encoders :
            num_ftrs = self.model.fc.in_features
            self.num_ftrs = num_ftrs
            self.model.fc = Identity()
        
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = {}
        for k, v in ckpt.items():
            if 'main_encoder' in k:
                k = k.replace('main_encoder.', '')
                state_dict[k] = v
        msg = self.model.load_state_dict(state_dict, strict=False)
        print(msg)
        print(colored("[FrankMocapHandFusionEncoder] Loaded pretrained weights from {}".format(ckpt_path), 'cyan'))

        # remove avg pool
        pooling_method = "avg" # or "max"
        if pooling_method == "max": # experiment results are not good
            self.model.avgpool = nn.AdaptiveMaxPool2d(1)
        elif pooling_method == "avg":
            self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        else:
            raise NotImplementedError
        print(colored("[FrankMocapHandEncoder] Using {} pooling".format(pooling_method), 'cyan'))
        

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        x = self.model.maxpool(x) # bx64x56x56

        feature_low_level = x.clone()
        feature_low_level = self.model.maxpool(feature_low_level) # bx64x28x28
        # visualize_feature_map_CHW(feature_low_level[0], "feature_low_level.png")
        feature_low_level = feature_low_level.mean(dim=1) # bx28x28
        feature_low_level = feature_low_level.flatten(start_dim=1) # bx784
        

        # visualize_feature_map_CHW(x[0], "maxpool.png")
        x = self.model.layer1(x) # bx256x56x56
        # visualize_feature_map_CHW(x[0], "layer1.png")
        x = self.model.layer2(x) # bx512x28x28
        # visualize_feature_map_CHW(x[0], "layer2.png")
        x = self.model.layer3(x) # bx1024x14x14
        # visualize_feature_map_CHW(x[0], "layer3.png")
        x = self.model.layer4(x) # bx2048x7x7
        # visualize_feature_map_CHW(x[0], "layer4.png")
        x = self.model.avgpool(x) # bx2048x1x1
        x = x.flatten(start_dim=1) # bx2048

        # concat high level and low level features
        x = torch.cat([x, feature_low_level], dim=1) # bx(2048+784)=2832

        return x

    def get_transform(self):
        trans = nn.Sequential(transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225]))
        return trans
    
    def get_features(self, x):
        with torch.no_grad():
            z = self.forward(x)
        return z.cpu().data.numpy() ### Can't store everything in GPU :/
    


class FrankMocapBodyEncoder(nn.Module):
    def __init__(self, encoder_type, ckpt_path):
        super(FrankMocapBodyEncoder, self).__init__()
        self.encoder_type = encoder_type
        if self.encoder_type in _encoders :
            self.model = _encoders[encoder_type](pretrained=True)
        else :
            print("Please enter a valid encoder type")
            raise Exception
        
        self.latent_dim = LATENT_DIM[encoder_type]
        for param in self.model.parameters():
            param.requires_grad = False
        if self.encoder_type in _encoders :
            num_ftrs = self.model.fc.in_features
            self.num_ftrs = num_ftrs
            self.model.fc = Identity()
        
        ckpt = torch.load(ckpt_path, map_location='cpu')
        msg = self.model.load_state_dict(ckpt['model'], strict=False)
        print(msg)
        print(colored("[FrankMocapBodyEncoder] Loaded pretrained weights from {}".format(ckpt_path), 'cyan'))

    def forward(self, x):
        x = self.model(x)
        return x

    def get_transform(self):
        trans = nn.Sequential(transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225]))
        return trans
    
    def get_features(self, x):
        with torch.no_grad():
            z = self.model(x)
        return z.cpu().data.numpy() ### Can't store everything in GPU :/
    

class FrankMocapHandJointEncoder(nn.Module):
    def __init__(self, encoder_type, ckpt_path):
        super(FrankMocapHandJointEncoder, self).__init__()
        self.encoder_type = encoder_type
        use_additional_resnet = False
        self.use_additional_resnet = use_additional_resnet
        if self.encoder_type in _encoders :
            self.hand_encoder = _encoders[encoder_type](pretrained=True)
            if use_additional_resnet:
                self.resnet = _encoders[encoder_type](pretrained=True)
        else :
            print("Please enter a valid encoder type")
            raise Exception
        if use_additional_resnet:
            self.resnet.fc = Identity()
        self.hand_encoder.fc = Identity()
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2048, 1024))

        
        self.latent_dim = int ( 1024 + 2048 * use_additional_resnet )

        for param in self.hand_encoder.parameters():
            param.requires_grad = False
        for param in self.fc.parameters():
            param.requires_grad = False
        if use_additional_resnet:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = {}
        for k, v in ckpt.items():
            if 'main_encoder' in k:
                k = k.replace('main_encoder.', '')
                state_dict[k] = v
        msg = self.hand_encoder.load_state_dict(state_dict, strict=False)
        msg2 = self.fc.load_state_dict(state_dict, strict=False)
        print("feature extract:", msg)
        print(colored("[FrankMocapHandJointEncoder] Loaded pretrained weights from {}".format(ckpt_path), 'cyan'))

    def forward(self, x):
        x = self.hand_encoder(x)
        x = self.fc(x)
        if self.use_additional_resnet:
            visual_state = self.resnet(x)
            x = torch.cat([x, visual_state], dim=1)
        return x

    def get_transform(self):
        trans = nn.Sequential(transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225]))
        return trans
    
    def get_features(self, x):
        with torch.no_grad():
            z = self.forward(x)
        return z.cpu().data.numpy() ### Can't store everything in GPU :/
 

class FrankMocapHandStateEncoder(nn.Module):
    def __init__(self, encoder_type, ckpt_path):
        super(FrankMocapHandStateEncoder, self).__init__()


        self.hand_encoder = _encoders["resnet50"](pretrained=True)
        self.resnet = _encoders["resnet34"](pretrained=True)


        self.resnet.fc = Identity()
        self.hand_encoder.fc = Identity()
        self.hand_state_encoder = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),)

        
        self.latent_dim = 1024 + 512

        for param in self.hand_encoder.parameters():
            param.requires_grad = False
        for param in self.hand_state_encoder.parameters():
            param.requires_grad = False
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # load hand feature encoder
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = {}
        for k, v in ckpt.items():
            if 'main_encoder' in k:
                k = k.replace('main_encoder.', '')
                state_dict[k] = v
        msg = self.hand_encoder.load_state_dict(state_dict, strict=False)
        print("feature extract:", msg)

        # load hand state encoder
        state_dict = {}
        state_dict['1.weight'] =ckpt['main_encoder.fc1.weight']
        state_dict['1.bias'] =ckpt['main_encoder.fc1.bias']
        state_dict['3.weight'] =ckpt['feat_encoder.1.weight']
        state_dict['3.bias'] =ckpt['feat_encoder.1.bias']
        msg = self.hand_state_encoder.load_state_dict(state_dict, strict=False)
        print("hand state encoder:", msg)
        
        print(colored("[FrankMocapHandStateEncoder] Loaded pretrained weights from {}".format(ckpt_path), 'cyan'))

    def forward(self, x):
        hand_feat = self.hand_encoder(x)
        hand_state = self.hand_state_encoder(hand_feat)
        visual_state = self.resnet(x)
        x = torch.cat([hand_state, visual_state], dim=1)
        return x

    def get_transform(self):
        trans = nn.Sequential(transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225]))
        return trans
    
    def get_features(self, x):
        with torch.no_grad():
            z = self.forward(x)
        return z.cpu().data.numpy() ### Can't store everything in GPU :/


class ResNetFT(nn.Module):
    def __init__(self, encoder_type, ckpt_path):
        super(ResNetFT, self).__init__()
        self.encoder_type = encoder_type
        if self.encoder_type in _encoders :
            self.model = _encoders[encoder_type](pretrained=True)
            # self.model_origin = _encoders[encoder_type](pretrained=True)
        else :
            print("Please enter a valid encoder type")
            raise Exception
        
        self.latent_dim = LATENT_DIM[encoder_type] * 2
        for param in self.model.parameters():
            param.requires_grad = False
        if self.encoder_type in _encoders :
            num_ftrs = self.model.fc.in_features
            self.num_ftrs = num_ftrs
            self.model.fc = Identity()
            # self.model_origin.fc = Identity()
        
        self.model.load_state_dict(torch.load(ckpt_path))

    def forward(self, x):
        # latent_origin = self.model_origin(x)
        latent = self.model(x)
        return latent

    def get_transform(self):
        trans = nn.Sequential(transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225]))
        return trans
    
    def get_features(self, x):
        with torch.no_grad():
            z = self.forward(x)
        return z.cpu().data.numpy() ### Can't store everything in GPU :/
 


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
