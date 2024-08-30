# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from fastreid.utils.one_hot import one_hot


class Circle3(nn.Module):
    def __init__(self, cfg, in_feat, num_classes):
        super().__init__()
        self.in_feat = in_feat
        self._num_classes = num_classes
        self._s = cfg.MODEL.HEADS.SCALE
        self._m = cfg.MODEL.HEADS.MARGIN

        self.weight = Parameter(torch.Tensor(num_classes, in_feat))
        self.weight2 = Parameter(torch.Tensor(num_classes, in_feat))
        self.weight3 = Parameter(torch.Tensor(num_classes, in_feat))

    def base_forward(self, weight, features, targets):
        sim_mat = F.linear(F.normalize(features), F.normalize(weight))
        alpha_p = F.relu(-sim_mat.detach() + 1 + self._m)
        alpha_n = F.relu(sim_mat.detach() + self._m)
        delta_p = 1 - self._m
        delta_n = self._m

        s_p = self._s * alpha_p * (sim_mat - delta_p)
        s_n = self._s * alpha_n * (sim_mat - delta_n)

        targets = one_hot(targets, self._num_classes)

        pred_class_logits = targets * s_p + (1.0 - targets) * s_n

        return pred_class_logits

    def forward(self, features, targets, sub=None):
        p1 = self.base_forward(self.weight, features, targets)
        p2 = self.base_forward(self.weight2, features[sub==0], targets[sub==0])
        p3 = self.base_forward(self.weight3, features[sub==1], targets[sub==1])
        t1 = targets
        t2 = targets[sub==0]
        t3 = targets[sub==1]

        pred_class_logits = torch.cat((p1, p2, p3), dim=0)
        targets = torch.cat((t1, t2, t3), dim=0)

        return pred_class_logits, targets

    def extra_repr(self):
        return 'in_features={}, num_classes={}, scale={}, margin={}'.format(
            self.in_feat, self._num_classes, self._s, self._m
        )
