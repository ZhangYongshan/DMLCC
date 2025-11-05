# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from convnext import Block, LayerNorm


# 经过每一个stage后，不需要对输入特征进行空间上的降维，即减少了原convnext的后面三个下采样层
class ConvNeXtIsotropic(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
        https://arxiv.org/pdf/2201.03545.pdf
        Isotropic ConvNeXts (Section 3.3 in paper)

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depth (tuple(int)): Number of blocks. Default: 18.
        dims (int): Feature dimension. Default: 384
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depth=2, dim=384, drop_path_rate=0.,
                 layer_scale_init_value=0, head_init_scale=1.,
                 ):
        super().__init__()
        #slic使用的代码
        # self.stem = nn.Conv2d(in_chans, dim, kernel_size=3, stride=1,padding=1)
        #原方法使用的代码
        self.stem = nn.Conv2d(in_chans, dim, kernel_size=3, stride=1)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[Block(dim=dim, drop_path=dp_rates[i],
                                            layer_scale_init_value=layer_scale_init_value)
                                      for i in range(depth)])
    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return x
        # return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        return x


class Re_ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depth=2, dim=384, drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[Block(dim=dim, drop_path=dp_rates[i],
                                            layer_scale_init_value=layer_scale_init_value)
                                      for i in range(depth)])
        self.upstem = nn.ConvTranspose2d(dim, in_chans, kernel_size=3, stride=1)

    def forward_features(self, x):
        x = self.blocks(x)
        x = self.upstem(x)
        return x
        # return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        return x


# @register_model
# def convnext_isotropic_small(pretrained=False, **kwargs):
#     model = ConvNeXtIsotropic(depth=18, dim=384, **kwargs)
#     if pretrained:
#         url = 'https://dl.fbaipublicfiles.com/convnext/convnext_iso_small_1k_224_ema.pth'
#         checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
#         model.load_state_dict(checkpoint["model"])
#     return model
#
#
# @register_model
# def convnext_isotropic_base(pretrained=False, **kwargs):
#     model = ConvNeXtIsotropic(depth=18, dim=768, **kwargs)
#     if pretrained:
#         url = 'https://dl.fbaipublicfiles.com/convnext/convnext_iso_base_1k_224_ema.pth'
#         checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
#         model.load_state_dict(checkpoint["model"])
#     return model
#
#
# @register_model
# def convnext_isotropic_large(pretrained=False, **kwargs):
#     model = ConvNeXtIsotropic(depth=36, dim=1024, **kwargs)
#     if pretrained:
#         url = 'https://dl.fbaipublicfiles.com/convnext/convnext_iso_large_1k_224_ema.pth'
#         checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
#         model.load_state_dict(checkpoint["model"])
#     return model
