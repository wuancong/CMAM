import torch.nn as nn
import math
import sys
import os.path as osp
import torch
from .build import BACKBONE_REGISTRY
import logging
from fastreid.utils.checkpoint import get_unexpected_parameters_message, get_missing_parameters_message

logger = logging.getLogger(__name__)
class SwitchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


__all__ = ['ResNetV2SN', 'resnetv2sn18', 'resnetv2sn34', 'resnetv2sn50', 'resnetv2sn101',
           'resnetv2sn152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, using_moving_average=True, using_bn=True):
        super(BasicBlock, self).__init__()
        self.sn1 = SwitchNorm2d(inplanes, using_moving_average=using_moving_average, using_bn=using_bn)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.sn2 = SwitchNorm2d(planes, using_moving_average=using_moving_average, using_bn=using_bn)
        self.conv2 = conv3x3(planes, planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.sn1(x)
        out_preact = self.relu(out)
        out = self.conv1(out_preact)

        out = self.sn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(out_preact)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, using_moving_average=False, using_bn=True):
        super(Bottleneck, self).__init__()
        self.sn1 = SwitchNorm2d(inplanes, using_moving_average=using_moving_average, using_bn=using_bn)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.sn2 = SwitchNorm2d(planes, using_moving_average=using_moving_average, using_bn=using_bn)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.sn3 = SwitchNorm2d(planes, using_moving_average=using_moving_average, using_bn=using_bn)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.sn1(x)
        out_preact = self.relu(out)
        out = self.conv1(out_preact)

        out = self.sn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.sn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(out_preact)

        out += residual

        return out


class ResNetV2SN(nn.Module):

    def __init__(self, block, layers, num_classes=1000, using_moving_average=True, using_bn=True):
        self.inplanes = 64
        self.using_moving_average=using_moving_average
        self.using_bn = using_bn
        super(ResNetV2SN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.sn1 = SwitchNorm2d(64, using_moving_average=self.using_moving_average, using_bn=self.using_bn)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.sn_out = SwitchNorm2d(512 * block.expansion, using_moving_average=self.using_moving_average, using_bn=self.using_bn)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.drouput = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SwitchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,using_moving_average=self.using_moving_average, using_bn=self.using_bn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, using_moving_average=self.using_moving_average, using_bn=self.using_bn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.sn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.sn_out(x)

        return x


def resnetv2sn18(**kwargs):
    """Constructs a ResNetV2SN-18 model using switchable normalization.
    """
    model = ResNetV2SN(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnetv2sn34(**kwargs):
    """Constructs a ResNetV2SN-34 model using switchable normalization.
    """
    model = ResNetV2SN(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnetv2sn50(**kwargs):
    """Constructs a ResNetV2SN-50 model using switchable normalization.
    """
    model = ResNetV2SN(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnetv2sn101(**kwargs):
    """Constructs a ResNetV2SN-101 model using switchable normalization.
    """
    model = ResNetV2SN(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnetv2sn152(**kwargs):
    """Constructs a ResNetV2SN-152 model using switchable normalization.
    """
    model = ResNetV2SN(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
from collections import OrderedDict
KNOWN_MODELS = OrderedDict([
    (50, lambda *a, **kw: resnetv2sn50(**kw)),
    (101, lambda *a, **kw: resnetv2sn101(**kw)),
    (152, lambda *a, **kw: resnetv2sn152(**kw))
])
@BACKBONE_REGISTRY.register()
def build_swi_backbone(cfg):
    import numpy as np
    """
    Create a ResNest instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """

    # fmt: off
    pretrain = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    depth = cfg.MODEL.BACKBONE.DEPTH

    model = KNOWN_MODELS[depth]()
    # print( 'pretrain', cfg.MODEL.BACKBONE.PRETRAIN,pretrain )
    if pretrain:
        state_dict = torch.load(pretrain_path)['state_dict']  # ibn-net
        # print( 'keys', list(state_dict.keys())[:10] )
        # print( 'model keys', list(model.state_dict().keys())[:10])
        # assert False
        # Remove module in name
        new_state_dict = {}
        for k in state_dict:
            new_k = k
            # print( 'k', k )
            if 'module' in new_k:
                new_k = new_k[7:]
            if new_k in model.state_dict() and (model.state_dict()[new_k].shape == state_dict[k].shape):
                new_state_dict[new_k] = state_dict[k]
        state_dict = new_state_dict
        logger.info(f"Loading pretrained model from {pretrain_path}")
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            logger.info(
                get_missing_parameters_message(incompatible.missing_keys)
            )
        if incompatible.unexpected_keys:
            logger.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )

    return model