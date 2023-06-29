import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models.resnet as resnet

from termcolor import colored

class TimeCorrEncoder(nn.Module):

    def __init__(self,ckpt_path):
        super(TimeCorrEncoder, self).__init__()

        model_state = torch.load(ckpt_path, map_location='cpu')['state_dict']

        net = resnet.resnet50()
        net_state = net.state_dict()

        # load net
        for k in [k for k in model_state.keys() if 'encoderVideo' in k]:
            kk = k.replace('module.encoderVideo.', '')
            tmp = model_state[k]
            if net_state[kk].shape != model_state[k].shape and net_state[kk].dim() == 4 and model_state[k].dim() == 5:
                tmp = model_state[k].squeeze(2)
            if len(net_state[kk].shape) != 0:
                net_state[kk][:] = tmp[:]
            else:
                net_state[kk] = tmp
            
        net.load_state_dict(net_state)

        afterconv1 = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        relu_layer = nn.ReLU(inplace=True)

        afterconv1_weight = model_state['module.afterconv1.weight']
        afterconv1_weight = afterconv1_weight.squeeze(2)

        x = afterconv1.state_dict()
        x['weight'][:] = afterconv1_weight[:]

        layers = list(net.children())[:-3]
        layers.append(afterconv1)
        # layers.append(relu_layer)

        net = nn.Sequential(*layers)
        self.encoder = net
        # self.latent_dim = 512
        self.latent_dim = 15*15*512

    def forward(self, x):
        x = self.encoder(x)
        # flat
        x = x.view(-1, 15*15*512)
        # downsample
        # x = nn.AvgPool2d(15)(x).view(-1, 512)
        return x

    def get_transform(self):
        # need to align pretraining progress !!!
        trans = nn.Sequential(transforms.Resize(256),
                                transforms.CenterCrop(240),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225]))
        return trans
    
    def get_features(self, x):
        with torch.no_grad():
            z = self.forward(x)
        return z.cpu().data.numpy() ### Can't store everything in GPU :/
