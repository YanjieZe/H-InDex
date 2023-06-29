import torch
import torch.nn as nn
import torch.nn.functional as F

class TestTimeBatchNorm2D(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.01, affine=True, track_running_stats=True):
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

