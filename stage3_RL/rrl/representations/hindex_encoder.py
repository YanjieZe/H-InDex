import torch
import torch.nn as nn
import torchvision.transforms as transforms
import omegaconf

from .TTP.imm_joint_model import get_pose_net

from termcolor import colored, cprint
import rrl.modules as rrl_modules
import copy

def replace_bn_recursively(module, momentum=0.01):
    for name, child_module in module.named_children():
        if isinstance(child_module, nn.BatchNorm2d):
            new_bn = rrl_modules.TestTimeBatchNorm2D(child_module.num_features, child_module.eps, momentum, child_module.affine, child_module.track_running_stats)
            new_bn.running_mean = child_module.running_mean
            new_bn.running_var = child_module.running_var
            new_bn.weight = child_module.weight
            new_bn.bias = child_module.bias
            setattr(module, name, new_bn)
            # cprint(f"Replace {name} with TestTimeBatchNorm2D", "yellow")
        else:
            replace_bn_recursively(child_module, momentum)

class HInDexEncoderBN(nn.Module):

    def __init__(self, cfg_path, ckpt_path, bn_momentum=0.01):
        super(HInDexEncoderBN, self).__init__()
        cfg = omegaconf.OmegaConf.load(cfg_path)
        self.model = get_pose_net(cfg, is_train=False, is_finetune=False, freeze_bn=False)
        self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load(ckpt_path), strict=True)

        self.model = self.model.module.pose_net

        print(colored("[HInDexEncoderBN] Encoder loaded from {}".format(ckpt_path), "cyan"))

        self.latent_dim = 2048
        self.cuda()

        # use test time batch norm
        # use_test_time_bn = True
        # if use_test_time_bn:
        #     replace_bn_recursively(self.pose_net, bn_momentum)
        #     cprint("Use test time batch norm with momentum {}".format(bn_momentum), "yellow")
        
        # use test time batch norm
        use_test_time_bn = True if bn_momentum > 0.0 else False
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
        self.running_mean = []
        self.running_var = []
        self.count = 0


    def forward(self, x):
        x = self.model.extract_feature(x)
        visualize_bn = False
        if visualize_bn:
            # record running mean and var
            self.running_mean.append(self.model.bn1.running_mean.mean().cpu().data.numpy())
            self.running_var.append(self.model.bn1.running_var.mean().cpu().data.numpy())
            self.count += 1
            if self.count == 1000:
                # visualize the curve
                import matplotlib.pyplot as plt
                import numpy as np
                plt.figure()
                # plot var as shadow
                plt.plot(np.arange(len(self.running_mean)), self.running_mean)
                plt.fill_between(np.arange(len(self.running_mean)), np.array(self.running_mean) - np.array(self.running_var), np.array(self.running_mean) + np.array(self.running_var), alpha=0.5)
                plt.savefig("bn.png")
                    
                import ipdb; ipdb.set_trace()


        return x

    def get_transform(self):
        trans = nn.Sequential(transforms.Resize(256),
                                transforms.CenterCrop(224),)
        return trans
    
    def get_features(self, x):
        with torch.no_grad():
            z = self.forward(x)
        return z.cpu().data.numpy() ### Can't store everything in GPU :/
