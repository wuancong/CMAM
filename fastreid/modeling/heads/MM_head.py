# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from fastreid.layers import *
from fastreid.utils.weight_init import weights_init_kaiming, weights_init_classifier
from .build import REID_HEADS_REGISTRY
from torch import nn
import torch
import torch.nn.functional as F
from fastreid.modeling.losses.cross_entroy_loss import CrossEntropyLoss

@REID_HEADS_REGISTRY.register()
class MMHead(nn.Module):
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
        else:
            raise KeyError(f"{cls_type} is invalid, please choose from "
                           f"'linear', 'arcface' and 'circle'.")

        self.classifier.apply(weights_init_classifier)

        self.build_MM(cfg, in_feat, num_classes)
    def build_MM(self, cfg, in_feat, num_classes):
        self.visible_classifier = Circle(cfg, in_feat, num_classes)
        self.infrared_classifier = Circle(cfg, in_feat, num_classes)

        self.visible_classifier.apply(weights_init_classifier)
        self.infrared_classifier.apply(weights_init_classifier)

        self.visible_classifier_ = Circle(cfg, in_feat, num_classes)
        self.visible_classifier_.weight.requires_grad_(False)
        self.visible_classifier_.weight.data = self.visible_classifier.weight.data

        self.infrared_classifier_ = Circle(cfg, in_feat, num_classes)
        self.infrared_classifier_.weight.requires_grad_(False)
        self.infrared_classifier_.weight.data = self.infrared_classifier.weight.data

        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        self.id_loss = CrossEntropyLoss(cfg)
        self.weight_sid =  cfg.MODEL.LOSSES.MM.SIDWEI
        self.weight_KL =  cfg.MODEL.LOSSES.MM.KLWEI
        self.update_rate = cfg.MODEL.LOSSES.MM.LAMBDA

    def MMLoss(self, feat, sub, labels):
        loss = 0
        logits_v = self.visible_classifier(feat[sub == 0], labels[sub==0])
        v_cls_loss = self.id_loss(logits_v.float(), None, labels[sub == 0])['loss_cls']
        loss += v_cls_loss * self.weight_sid
        logits_i = self.infrared_classifier(feat[sub == 1], labels[sub==1])
        i_cls_loss = self.id_loss(logits_i.float(), None, labels[sub == 1])['loss_cls']
        loss += i_cls_loss * self.weight_sid

        logits_m = torch.cat([logits_v, logits_i], 0).float()
        with torch.no_grad():
            self.infrared_classifier_.weight.data = self.infrared_classifier_.weight.data * (1 - self.update_rate) \
                                                    + self.infrared_classifier.weight.data * self.update_rate
            self.visible_classifier_.weight.data = self.visible_classifier_.weight.data * (1 - self.update_rate) \
                                                   + self.visible_classifier.weight.data * self.update_rate

            logits_v_ = self.infrared_classifier_(feat[sub == 0], labels[sub==0])
            logits_i_ = self.visible_classifier_(feat[sub == 1], labels[sub==1])

            logits_m_ = torch.cat([logits_v_, logits_i_], 0).float()
        logits_m = F.softmax(logits_m, 1)
        logits_m_ = F.log_softmax(logits_m_, 1)
        mod_loss = self.KLDivLoss(logits_m_, logits_m)

        loss += mod_loss * self.weight_KL + (v_cls_loss + i_cls_loss) * self.weight_sid
        return loss

    def forward(self, features, targets=None, bn_precise=False, sub=None):
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
        except TypeError: pred_class_logits = self.classifier(bn_feat, targets)

        if self.neck_feat == "before":  feat = global_feat[..., 0, 0]
        elif self.neck_feat == "after": feat = bn_feat
        else:
            raise KeyError("MODEL.HEADS.NECK_FEAT value is invalid, must choose from ('after' & 'before')")

        mm_loss = self.MMLoss(bn_feat, sub, targets)
        return pred_class_logits, feat, targets, mm_loss
