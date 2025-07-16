# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from iod.layers import Conv2d, ShapeSpec, get_norm
from iod.utils.registry import Registry

ROI_BOX_HEAD_REGISTRY = Registry("ROI_BOX_HEAD")
ROI_BOX_HEAD_REGISTRY.__doc__ = """
Registry for box heads, which make box predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


@ROI_BOX_HEAD_REGISTRY.register()
class FastRCNNConvFCHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and
    several fc layers (each followed by relu).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__()

        # fmt: off
        num_conv   = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim   = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc     = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim     = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        norm       = cfg.MODEL.ROI_BOX_HEAD.NORM
        # fmt: on
        assert num_conv + num_fc > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not norm,
                norm=get_norm(norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))
        return x

    @property
    def output_size(self):
        return self._output_size


@ROI_BOX_HEAD_REGISTRY.register()
class ATTENTION(nn.Module):
    def __init__(self, cfg, input_shape: ShapeSpec, reduction=16):
        super(ATTENTION, self).__init__()
        self._output_size = (
            input_shape.channels,
            input_shape.height,
            input_shape.width,
        )
        num_fc     = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim     = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        norm       = cfg.MODEL.ROI_BOX_HEAD.NORM

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc_1 = nn.Conv2d( self._output_size[0],  self._output_size[0] // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc_2 = nn.Conv2d( self._output_size[0] // reduction, self._output_size[0], kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim


    def forward(self, x):
        original = x
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        x = x_avg + x_max
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.sigmoid(x)
        # x = original * x
        x = original + x
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))
        return x
    @property
    def output_size(self):
        return self._output_size



@ROI_BOX_HEAD_REGISTRY.register()
class GCNET(nn.Module):
    def __init__(self, cfg, input_shape: ShapeSpec, reduction=16):
        super(GCNET, self).__init__()
        self._output_size = (
            input_shape.channels,
            input_shape.height,
            input_shape.width,
        )
        num_fc     = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim     = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        norm       = cfg.MODEL.ROI_BOX_HEAD.NORM

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc_1 = nn.Conv2d( self._output_size[0],  self._output_size[0] // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc_2 = nn.Conv2d( self._output_size[0] // reduction, self._output_size[0], kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.conv_WK = nn.Conv2d(self._output_size[0], 1, kernel_size=1, bias=True)
        self.conv_WV1 = nn.Conv2d(self._output_size[0], self._output_size[0]//reduction, kernel_size=1, bias=True)
        self.LayerNorm = nn.LayerNorm([self._output_size[0]//reduction, 1, 1])
        self.conv_WV2 = nn.Conv2d(self._output_size[0] // reduction, self._output_size[0], kernel_size=1, bias=True)
        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim


    def forward(self, x):
        N, C, H, W = x.shape
        hm = F.softmax(self.conv_WK(x).view(N, -1, 1, 1), dim=1)
        x_c = x.view(N, C, -1)
        #a = hm.unsqueeze(1)
        y = torch.matmul(x_c.unsqueeze(1), hm.unsqueeze(1).squeeze(4)).view(N, C, 1, 1)
        y = self.conv_WV1(y)
        y = self.LayerNorm(y)
        y = F.relu(y, inplace=True)
        y = self.conv_WV2(y)
        y = y + x
        if len(self.fcs):
            if y.dim() > 2:
                y = torch.flatten(y, start_dim=1)
            for layer in self.fcs:
                y = F.relu(layer(y))
        # y = F.relu(y, inplace=True)
        return y
    
    @property
    def output_size(self):
        return self._output_size


def build_box_head(cfg, input_shape):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_BOX_HEAD.NAME
    return ROI_BOX_HEAD_REGISTRY.get(name)(cfg, input_shape)
