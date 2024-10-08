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
from fastreid.modeling.losses.cross_entroy_loss import CrossEntropyLoss
from fastreid.utils.one_hot import one_hot


class CircleMeta(nn.Module):
    def __init__(self, cfg, in_feat, num_classes):
        super().__init__()
        self.in_feat = in_feat
        self._num_classes = num_classes
        self._s = cfg.MODEL.HEADS.SCALE
        self._m = cfg.MODEL.HEADS.MARGIN

        self.weight = Parameter(torch.Tensor(num_classes, in_feat))
        self.loss = CrossEntropyLoss(cfg)
        self.cfg = cfg
    def norm_froward(self, features, targets, weight):
        sim_mat = F.linear(F.normalize(features), weight)
        alpha_p = F.relu(-sim_mat.detach() + 1 + self._m)
        alpha_n = F.relu(sim_mat.detach() + self._m)
        delta_p = 1 - self._m
        delta_n = self._m

        s_p = self._s * alpha_p * (sim_mat - delta_p)
        s_n = self._s * alpha_n * (sim_mat - delta_n)

        targets = one_hot(targets, self._num_classes)

        pred_class_logits = targets * s_p + (1.0 - targets) * s_n

        return pred_class_logits

    def forward(self, features, targets, sub):
        logits = self.norm_froward(features[sub==1], targets[sub==1], self.weight)
        loss = self.loss(logits, None, targets[sub==1])['loss_cls']
        grads = torch.autograd.grad(loss, self.weight,
                                    create_graph=True)
        new_weight = self.weight.data - self.cfg.META.LR * grads[0]
        new_logits = self.norm_froward(features, targets, new_weight)

        return new_logits

    def extra_repr(self):
        return 'in_features={}, num_classes={}, scale={}, margin={}'.format(
            self.in_feat, self._num_classes, self._s, self._m
        )
