import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

from termcolor import colored

class HandObjectDetectExtractor(nn.Module):

    def __init__(self, ckpt_path):
        super(HandObjectDetectExtractor, self).__init__()
        resnet = models.resnet101(pretrained=True)
        self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
            resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)

        self.RCNN_top = nn.Sequential(resnet.layer4)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        """
        this part is not used yet. maybe explore how to use later
        """
        self.n_classes = 3
        self.RCNN_cls_score = nn.Linear(2048, self.n_classes)
        self.RCNN_bbox_pred = nn.Linear(2048, 4 * self.n_classes)
    
        self.latent_dim = 2048

        ckpt = torch.load(ckpt_path, map_location='cpu')
        msg = self.load_state_dict(ckpt['model'], strict=False)
        print(colored(msg, 'cyan'))
        print(colored('[HandObjectDetectExtractor] Loaded checkpoint from {}'.format(ckpt_path), 'cyan'))


    def forward(self, x):
        base_feat = self.RCNN_base(x) # b, 1024, 14, 14
        """"
        actually, original rcnn should do roi pooling here, but we don't do this
        """
        final_feat = self.RCNN_top(base_feat) # b, 2048, 7, 7

        # avg pooling
        final_feat = self.avg_pool(final_feat) # b, 2048, 1, 1

        # flatten
        final_feat = final_feat.view(final_feat.size(0), -1) # b, 2048

        return final_feat

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
