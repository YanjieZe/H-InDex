import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from termcolor import colored, cprint
import copy
class TestTimeBatchNorm2D(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        # 如果在推理模式，强制更新 running_mean 和 running_var
        if not self.training and self.track_running_stats:
            with torch.no_grad():
                exponential_average_factor = 0.0
                if self.momentum is None:  # 使用累积移动平均
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # 使用指数移动平均
                    exponential_average_factor = self.momentum
                self.num_batches_tracked += 1
                
                # 更新 running_mean 和 running_var
                self.running_mean = (1 - exponential_average_factor) * self.running_mean + exponential_average_factor * input.mean([0, 2, 3])
                self.running_var = (1 - exponential_average_factor) * self.running_var + exponential_average_factor * input.var([0, 2, 3], unbiased=False)
        return super().forward(input)
    
def get_frankmocap_hand_encoder(pretrain_path, img_size=224):
    r50 = models.resnet50(pretrained=True)
    r50.fc = nn.Identity()
    ckpt = torch.load(pretrain_path, map_location='cpu')
    state_dict = {}
    for k, v in ckpt.items():
        if 'main_encoder' in k:
            k = k.replace('main_encoder.', '')
            state_dict[k] = v
    msg = r50.load_state_dict(state_dict, strict=False)
    latent_dim = 2048
    print(msg)
    print(colored('[FrankMocapHandEncoder] Loaded pretrained weights from {}'.format(pretrain_path), 'cyan'))

    use_test_time_bn = True 
    new_model = copy.deepcopy(r50)
    momentum = 0.01
    if use_test_time_bn:
        for module_name, m in r50.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                # replace original batch norm with test time batch norm
                new_m = TestTimeBatchNorm2D(m.num_features, m.eps, momentum, m.affine, m.track_running_stats)
                # copy parameters
                new_m.running_mean = m.running_mean
                new_m.running_var = m.running_var
                new_m.weight = m.weight
                new_m.bias = m.bias
                # replace
                new_model._modules[module_name] = new_m
        cprint(f'[FrankMocapHandEncoder] Using test time batch norm with momentum {momentum}', 'cyan')
    r50 = new_model
    return r50, latent_dim
    

def get_r3m_encoder(pretrain_path, img_size=224):
    from r3m import load_r3m
    model_and_config_folder = pretrain_path
    rep = load_r3m("resnet50", model_and_config_folder=model_and_config_folder) # resnet18, resnet34
    print(colored(f"[R3MEncoder] Loaded R3M encoder from {model_and_config_folder}", "green"))
    rep.eval()
    rep = rep.module.convnet
    latent_dim = 2048
    return rep, latent_dim

