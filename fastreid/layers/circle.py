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


class Circle(nn.Module):
    def __init__(self, cfg, in_feat, num_classes):
        super().__init__()
        self.in_feat = in_feat
        self._num_classes = num_classes
        self._s = cfg.MODEL.HEADS.SCALE
        self._m = cfg.MODEL.HEADS.MARGIN

        self.weight = Parameter(torch.Tensor(num_classes, in_feat))

    def forward(self, features, targets, sub=None):
        ori_target = targets
        sim_mat = F.linear(F.normalize(features), F.normalize(self.weight))
        alpha_p = F.relu(-sim_mat.detach() + 1 + self._m)
        alpha_n = F.relu(sim_mat.detach() + self._m)
        delta_p = 1 - self._m
        delta_n = self._m

        s_p = self._s * alpha_p * (sim_mat - delta_p)
        s_n = self._s * alpha_n * (sim_mat - delta_n)

        targets = one_hot(targets, self._num_classes)

        pred_class_logits = targets * s_p + (1.0 - targets) * s_n

        return pred_class_logits, ori_target

    def norm_forward(self, features, targets, weight):
        sim_mat = F.linear(F.normalize(features), F.normalize(weight) )
        alpha_p = F.relu(-sim_mat.detach() + 1 + self._m)
        alpha_n = F.relu(sim_mat.detach() + self._m)
        delta_p = 1 - self._m
        delta_n = self._m

        s_p = self._s * alpha_p * (sim_mat - delta_p)
        s_n = self._s * alpha_n * (sim_mat - delta_n)

        targets = one_hot(targets, self._num_classes)

        pred_class_logits = targets * s_p + (1.0 - targets) * s_n
        return pred_class_logits, targets

    def fake_forward(self, features, targets, sub=None):
        sim_mat = F.linear(F.normalize(features), F.normalize(self.weight.detach()))
        alpha_p = F.relu(-sim_mat.detach() + 1 + self._m)
        alpha_n = F.relu(sim_mat.detach() + self._m)
        delta_p = 1 - self._m
        delta_n = self._m

        s_p = self._s * alpha_p * (sim_mat - delta_p)
        s_n = self._s * alpha_n * (sim_mat - delta_n)

        targets = one_hot(targets, self._num_classes)

        pred_class_logits = targets * s_p + (1.0 - targets) * s_n

        return pred_class_logits, targets

    def extra_repr(self):
        return 'in_features={}, num_classes={}, scale={}, margin={}'.format(
            self.in_feat, self._num_classes, self._s, self._m
        )


class DISENTANGLE(nn.Module):
    def __init__(self, cfg, in_feat=2048, num_classes=400):
        super().__init__()
        dim = in_feat
        self.dim = in_feat
        self.encoder = nn.Sequential(
            nn.Linear(in_feat, in_feat),
            nn.BatchNorm1d(in_feat),
            nn.GELU(),
            nn.Linear(in_feat, dim*2),
            nn.BatchNorm1d(dim*2),
            nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(dim*2, in_feat),
            nn.BatchNorm1d(in_feat),
            nn.GELU(),
            nn.Linear(in_feat, in_feat),
            nn.BatchNorm1d(in_feat),
        )

        self.cla1 = Circle(cfg, in_feat, num_classes=num_classes)
        self.cla2 = Circle(cfg, in_feat, num_classes=num_classes)
        self.cfg = cfg

    def forward(self, x1, x2, targets):
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)

        af1, bf1 = f1[:,:self.dim], f1[:,self.dim:]
        af2, bf2 = f2[:,:self.dim], f2[:,self.dim:]
        logita1,_ = self.cla1(af1, targets)
        logitb1,_ = self.cla2(bf1, targets)

        logita2, _ = self.cla1.fake_forward(af2, targets)
        logitb2, _ = self.cla2.fake_forward(bf2, targets)

        rx1 = self.decoder(f1)
        rx2 = self.decoder(f2)
        loss1 = torch.norm(rx1 - x1, dim=1).mean()
        loss2 = torch.norm(rx2 - x2, dim=1).mean()
        loss = (loss1 + loss2) * self.cfg.MODEL.LOSSES.DIS.RECSCALE
        return logita1, logitb1, logita2, logitb2, loss


class DISENTANGLE5(nn.Module):
    def __init__(self, cfg, in_feat=2048, num_classes=400):
        super().__init__()
        dim = in_feat
        self.dim = in_feat
        self.en1 = nn.Sequential(
            nn.Linear(in_feat, in_feat),
            nn.BatchNorm1d(in_feat),
            nn.GELU(),
            nn.Linear(in_feat, in_feat),
            nn.BatchNorm1d(in_feat),
            nn.LeakyReLU()
        )
        self.en2 = nn.Sequential(
            nn.Linear(in_feat, in_feat),
            nn.BatchNorm1d(in_feat),
            nn.GELU(),
            nn.Linear(in_feat, dim * 2),
            nn.BatchNorm1d(dim * 2),
            nn.LeakyReLU()
        )
        self.ch_coder = nn.Parameter(torch.zeros((3, in_feat)))

        self.decoer = nn.Sequential(
            nn.Linear(in_feat, in_feat),
            nn.BatchNorm1d(in_feat),
            nn.GELU(),
            nn.Linear(in_feat, in_feat),
            nn.BatchNorm1d(in_feat),
        )

        self.cfg = cfg
        self.cla1 = Circle(cfg, in_feat, num_classes=num_classes)
        self.cla2 = Circle(cfg, in_feat, num_classes=num_classes)
    def forward(self, fea, rec_fea, channels, targets):
        f1 = self.en1(fea)
        tmp = self.ch_coder[channels]
        f2 = tmp + f1
        enfs = self.en2(f2)
        f1, f2 = enfs[:,:self.dim], enfs[:,self.dim:]

        logit1, _ = self.cla1(f1, targets)
        logit2, _ = self.cla2(f2, targets)

        chrec_loss = torch.norm(f2 - rec_fea, 1).mean()
        derec_loss = torch.norm(fea - self.decoer(f2+f1), 1).mean()
        sims = (F.normalize(f1) * F.normalize(f2)).sum(dim=1)
        norm_loss = (sims + 1).sum()
        orth_loss = norm_loss * self.cfg.MODEL.LOSSES.DIS.ORTHSCALE
        rec_loss = (chrec_loss + derec_loss) * self.cfg.MODEL.LOSSES.DIS.RECSCALE

        return logit1, logit2, orth_loss, rec_loss







