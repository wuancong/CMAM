# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
from torch import nn

from fastreid.layers import GeneralizedMeanPoolingP, AdaptiveAvgMaxPool2d, FastGlobalAvgPool2d, AttentionPool2d
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_reid_heads
from fastreid.modeling.losses import reid_losses
from fastreid.utils.weight_init import weights_init_kaiming
from .build import META_ARCH_REGISTRY
import torch.nn.functional as F


@META_ARCH_REGISTRY.register()
class BASE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))
        self._cfg = cfg

        # backbone
        backbone = build_backbone(cfg)

        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )
        pool_layer = GeneralizedMeanPoolingP()

        # head
        in_feat = cfg.MODEL.HEADS.IN_FEAT
        num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        self.pool = self._build_pool_reduce(pool_layer)
        self.head = build_reid_heads(cfg, in_feat, num_classes, nn.Identity())

    @staticmethod
    def _build_pool_reduce(pool_layer):
        pool_reduce = nn.Sequential(
            pool_layer,
        )
        pool_reduce.apply(weights_init_kaiming)
        return pool_reduce

    @property
    def device(self):
        return self.pixel_mean.device

    def test_forward(self, images):
        features = self.backbone(images)  # (bs, 2048, 16, 8)
        pool_feat = self.pool(features).squeeze()
        test_feat = F.normalize(pool_feat, 1)
        return test_feat

    def train_forward(self, images, targets, camids, head_idx=1):
        # Training
        features = self.backbone(images)  # (bs, 2048, 24, 8)
        pool_feat = self.pool(features)
        if head_idx == 1:
            logits, pool_feat, _ = self.head(pool_feat, targets)
        elif head_idx == 2: # use the same head
            logits, pool_feat, _ = self.head(pool_feat, targets)
        return {'logits': logits, 'feats': pool_feat, 'targets': targets, 'camids': camids}

    def losses(self, outputs, prefix=''):
        return reid_losses(self._cfg, outputs['logits'], outputs['feats'], outputs['targets'], prefix)
