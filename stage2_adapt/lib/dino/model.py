import numpy as np
import torch
import torch.nn as nn

import dino.vision_transformer_flexible as vits
from termcolor import colored
import os

def download_file(url, local_path):
    import requests
    import os
    import shutil
    import sys
    import time

    if os.path.exists(local_path):
        print(f"File {local_path} already exists, skipping download")
        return local_path

    print(f"Downloading {url} to {local_path}")
    os.system(f"wget {url} -O {local_path}")

    print(f"Downloaded {url} to {local_path}")
    return local_path

class DINO(nn.Module):
    def __init__(self, pretrain_path=None):
        super().__init__()
        self.patch_size = 8
        self.feat_layer = 9
        self.high_res = False

        if self.patch_size == 16:
            self.model_name = "vit_base"
            self.stride = 8
            self.num_patches = 16
            self.padding = 5
            self.pretrain_path = download_file(
                "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
                "dino_vitbase16_pretrain.pth",
            ) if pretrain_path is None else pretrain_path
        elif self.patch_size == 8:
            self.model_name = "vit_small"
            self.stride = 4
            self.num_patches = 32
            self.padding = 2
            self.pretrain_path = download_file(
                "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth",
                "dino_deitsmall8_pretrain.pth",
            ) if pretrain_path is None else pretrain_path
        else:
            raise ValueError("ViT models only supported with patch sizes 8 or 16")

        if self.high_res:
            self.num_patches *= 2

        self.model = None
        self.load_model()

    def load_model(self):
        model = vits.__dict__[self.model_name](patch_size=self.patch_size)
        if not os.path.exists(self.pretrain_path):
            raise ValueError(f"Pretrained model not found at {self.pretrain_path}")
        else:
            print(colored(f"[DINO] Loading pretrained model from {self.pretrain_path}", "cyan"))
        state_dict = torch.load(self.pretrain_path, map_location="cpu")
        model.load_state_dict(state_dict)
        # model.to(device)
        model.eval()

        if self.high_res:
            model.patch_embed.proj.stride = (self.stride, self.stride)
            model.num_patches = self.num_patches ** 2
            model.patch_embed.patch_size = self.stride
            model.patch_embed.proj.padding = self.padding
        self.model = model

    def extract_features_and_attn(self, all_images):
        """
        A definition of relevant dimensions {all_b, nh, t, d}:
            image_size: Side length of input images (assumed square)
            all_b: The first dimension size of the input tensor - not necessarily
                the same as "batch size" in high-level script, as we assume that
                reference and target images are all flattened-then-concatenated
                along the batch dimension. With e.g. a batch size of 2, and 5 target
                images, 1 reference image; all_b = 2 * (5+1) = 12
            h: number of heads in ViT, e.g. 6
            t: number of items in ViT keys/values/tokens, e.g. 785 (= 28*28 + 1)
            d: feature dim in ViT, e.g. 64

        Args:
            all_images (torch.Tensor): shape (all_b, 3, image_size, image_size)
        Returns:
            features (torch.Tensor): shape (all_b, nh, t, d) e.g. (12, 6, 785, 64)
            attn (torch.Tensor): shape (all_b, nh, t, t) e.g. (12, 6, 785, 785)
            output_cls_tokens (torch.Tensor): shape (all_b, nh*d) e.g. (12, 384)
        """
        MAX_BATCH_SIZE = 50
        all_images_batch_size = all_images.size(0)
        c, img_h, img_w = all_images.shape[-3:]
        all_images = all_images.view(-1, c, img_h, img_w)
        with torch.no_grad():
            torch.cuda.empty_cache()

            if all_images_batch_size <= MAX_BATCH_SIZE:
                data = self.model.get_specific_tokens(all_images, layers_to_return=(9, 11))
                features = data[self.feat_layer]["k"]
                attn = data[11]["attn"]
                output_cls_tokens = data[11]["t"][:, 0, :]

            # Process in chunks to avoid CUDA out-of-memory
            else:
                num_chunks = np.ceil(all_images_batch_size / MAX_BATCH_SIZE).astype("int")
                data_chunks = []
                for i, ims_ in enumerate(all_images.chunk(num_chunks)):
                    data_chunks.append(self.model.get_specific_tokens(ims_, layers_to_return=(9, 11)))

                features = torch.cat([d[self.feat_layer]["k"] for d in data_chunks], dim=0)
                attn = torch.cat([d[11]["attn"] for d in data_chunks], dim=0)
                output_cls_tokens = torch.cat([d[11]["t"][:, 0, :] for d in data_chunks], dim=0)

        return features, attn, output_cls_tokens

    def forward(self, img, return_cls_attention=False):

        # imgnet normalize
        RANGE_IS_NEG1_TO_1 = False
        if RANGE_IS_NEG1_TO_1:
            img_norm = (img + 1) / 2
        else:
            img_norm = img
        mean, std = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img.device), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img.device)
        img_norm = (img_norm - mean) / std


        features, attn_matrix, output_cls_tokens = self.extract_features_and_attn(img_norm)

        features = features[:, :, 1:, :]
        features = features.permute(0, 1, 3, 2)
        bsz, nh, d, t = features.shape
        hf, wf = int(np.sqrt(t)), int(np.sqrt(t))
        features = features.reshape(bsz, d * nh, hf, wf)  # bsz, d*nh, h, w


        cls_attention = attn_matrix[..., 0, 1:].reshape(bsz, -1, hf, wf)

        

        # # visualize attention map
        # import matplotlib.pyplot as plt
        # for atten_idx in range(cls_attention.shape[1]):
        #     # show src img and attention in subplot
        #     fig, ax = plt.subplots(1, 2)
        #     img_show = img[idx].permute(1, 2, 0).cpu().numpy()
        #     # rgb to bgr
        #     img_show = img_show[..., ::-1]
        #     ax[0].imshow(img_show)
        #     ax[0].set_title('src img')
        #     ax[1].imshow(cls_attention[idx, atten_idx, :, :].cpu().numpy())
        #     ax[1].set_title(f'attention map {atten_idx}')
        #     plt.savefig(f'attention_matrix_{atten_idx}.png')
        
        # # mean all
        # ax[0].imshow(img_show)
        # ax[0].set_title('src img')
        # ax[1].imshow(cls_attention[idx, :, :, :].mean(dim=0).cpu().numpy())
        # ax[1].set_title(f'attention map mean')
        # plt.savefig(f'attention_matrix_mean.png')

        # # visualize features
        # import matplotlib.pyplot as plt
        # plt.imshow(features[idx].mean(dim=0).cpu().numpy())
        # plt.savefig(f'features.png')

        if return_cls_attention:
            return features, cls_attention

        return features
