# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from fastreid.layers import *
from fastreid.utils.weight_init import weights_init_kaiming, weights_init_classifier
from .build import REID_HEADS_REGISTRY
from torch import nn
import torch.nn.functional as F

@REID_HEADS_REGISTRY.register()
class BNneckHead(nn.Module):
    def __init__(self, cfg, in_feat, num_classes, pool_layer):
        super().__init__()
        self.neck_feat = cfg.MODEL.HEADS.NECK_FEAT
        self.pool_layer = pool_layer

        self.bnneck = get_norm(cfg.MODEL.HEADS.NORM, in_feat, cfg.MODEL.HEADS.NORM_SPLIT, bias_freeze=True)
        self.bnneck.apply(weights_init_kaiming)

        # identity classification layer
        cls_type = cfg.MODEL.HEADS.CLS_LAYER
        if cls_type == 'linear':    self.classifier = nn.Linear(in_feat, num_classes, bias=False)
        elif cls_type == 'arcface': self.classifier = Arcface(cfg, in_feat, num_classes)
        elif cls_type == 'circle':  self.classifier = Circle(cfg, in_feat, num_classes)
        elif cls_type == 'circle_meta': self.classifier = CircleMeta(cfg, in_feat, num_classes)
        elif cls_type == 'circle_3': self.classifier = Circle3(cfg, in_feat, num_classes)
        else:
            raise KeyError(f"{cls_type} is invalid, please choose from "
                           f"'linear', 'arcface' and 'circle'.")

        self.classifier.apply(weights_init_classifier)

    def forward(self, features, targets=None, bn_precise = False, sub=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        global_feat = self.pool_layer(features)
        bn_feat = self.bnneck(global_feat)
        bn_feat = bn_feat[..., 0, 0]
        if bn_precise:
            return
        # Evaluation
        if not self.training: return bn_feat
        # Training
        try:              pred_class_logits = self.classifier(bn_feat)
        except TypeError: pred_class_logits, targets = self.classifier(bn_feat, targets, sub=sub)

        if self.neck_feat == "before":  feat = global_feat[..., 0, 0]
        elif self.neck_feat == "after": feat = bn_feat
        else:
            raise KeyError("MODEL.HEADS.NECK_FEAT value is invalid, must choose from ('after' & 'before')")
        return pred_class_logits, feat, targets


    def norm_forward(self, features, targets, ori_bn_sta, new_bn_weight, cls_weight):
        globl_feat = self.pool_layer(features)
        bn_feat = F.batch_norm(globl_feat, self.bnneck.running_mean, self.bnneck.running_var, new_bn_weight, bias=self.bnneck.bias, training=True,
                               momentum=self.bnneck.momentum, eps=self.bnneck.eps)
        bn_feat = bn_feat[..., 0, 0]
        pred_class_logits, targets =  self.classifier.norm_forward(bn_feat, targets, cls_weight)
        return pred_class_logits, bn_feat, targets


