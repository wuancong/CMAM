# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy

import torch
from torch import nn

from fastreid.layers import GeneralizedMeanPoolingP, get_norm, AdaptiveAvgMaxPool2d, FastGlobalAvgPool2d, AttentionPool2d
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.backbones.resnet import Bottleneck
from fastreid.modeling.heads import build_reid_heads
from fastreid.modeling.losses import reid_losses, CrossEntropyLoss, CenterLoss
from fastreid.modeling.losses.metric_loss import ModTripletLoss
from fastreid.utils.weight_init import weights_init_kaiming
from .build import META_ARCH_REGISTRY
from .MGN_TRY import MAM
import torch.nn.functional as F
import numpy as np



class MMD(nn.Module):
    def __init__(self, cfg, infeat=2048, n_domain=2):
        super().__init__()
        self.centers = CenterLoss(n_domain, infeat)
        self.n_domain = n_domain
        self.cfg = cfg

    def forward(self, features, labels):
        cur_centers = []
        for i in range(self.n_domain):
            center = features[labels==i].mean(dim=0)
            cur_centers.append(center)
        cur_centers = torch.stack(cur_centers, 0)
        domain_labels = torch.arange(0, end=self.n_domain).cuda()
        loss = self.centers(cur_centers, domain_labels)

        loss += self.cfg.MODEL.LOSSES.MMD.INTER * F.mse_loss(self.centers.centers[0], self.centers.centers[1])
        return loss


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return  y

@META_ARCH_REGISTRY.register()
class BASE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))
        self._cfg = cfg
        self.with_ciconv = cfg.MODEL.BACKBONE.WITH_CICONV
        self.ciconv_invariant = cfg.MODEL.BACKBONE.CICONV_INVARIANT


        # backbone
        bn_norm = cfg.MODEL.BACKBONE.NORM
        num_splits = cfg.MODEL.BACKBONE.NORM_SPLIT
        with_se = cfg.MODEL.BACKBONE.WITH_SE
        with_ciconv = cfg.MODEL.BACKBONE.WITH_CICONV
        backbone = build_backbone(cfg)
        if with_ciconv:
            self.backbone = nn.Sequential(
                backbone.ciconv,
                backbone.conv1_for_ci,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4
            )
        else:
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

        pool_type = cfg.MODEL.HEADS.POOL_LAYER
        if pool_type == 'avgpool':      pool_layer = FastGlobalAvgPool2d()
        elif pool_type == 'maxpool':    pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempool':    pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == "avgmaxpool": pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == "identity":   pool_layer = nn.Identity()
        elif pool_type == 'attnpool': pool_layer = AttentionPool2d((24, 8), 2048, 16, 2048)
        else:
            raise KeyError(f"{pool_type} is invalid, please choose from "
                           f"'avgpool', 'maxpool', 'gempool', 'avgmaxpool' and 'identity'.")

        # head
        in_feat = cfg.MODEL.HEADS.IN_FEAT
        num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        # branch1
        self.b1 = nn.Identity()
        self.b1_pool = self._build_pool_reduce(pool_layer, bn_norm, num_splits, reduce_dim=in_feat, dropout=cfg.MUTUAL.DROPUOUT)
        self.b1_head = build_reid_heads(cfg, in_feat, num_classes, nn.Identity())

        # branch2 for third modality
        self.b2 = nn.Identity()
        self.b2_pool = self._build_pool_reduce(pool_layer, bn_norm, num_splits, reduce_dim=in_feat,
                                               dropout=cfg.MUTUAL.DROPUOUT)
        self.b2_head = build_reid_heads(cfg, in_feat, num_classes, nn.Identity())

        self.rgb_cams = [0, 1, 3, 4]
        self.ir_cams = [2, 5]


    @staticmethod
    def _build_pool_reduce(pool_layer, bn_norm, num_splits, input_dim=2048, reduce_dim=256, dropout=0.0):
        pool_reduce = nn.Sequential(
            pool_layer,
            # get_norm(bn_norm, reduce_dim, num_splits),
            # nn.ReLU(True),
        )
        pool_reduce.apply(weights_init_kaiming)
        return pool_reduce

    @property
    def device(self):
        return self.pixel_mean.device


    def test_forward(self, images):
        features = self.backbone(images)  # (bs, 2048, 16, 8)
        # branch1
        b1_feat = self.b1(features)
        b1_pool_feat = self.b1_pool(b1_feat)
        b1_pool_feat = self.b1_head(b1_pool_feat)

        b1_pool_feat = F.normalize(b1_pool_feat, 1)
        return b1_pool_feat


    def train_forward(self, images, targets, camids, fea_stack=True, head_idx=1):
        # Training
        features = self.backbone(images)  # (bs, 2048, 24, 8)
        # branch1
        b1_feat = self.b1(features)
        b1_pool_feat = self.b1_pool(b1_feat)
        b1_logits, b1_pool_feat, _ = self.b1_head(b1_pool_feat, targets)
        # branch2
        b2_feat = self.b2(features)
        b2_pool_feat = self.b2_pool(b2_feat)
        b2_logits, b2_pool_feat, _ = self.b2_head(b2_pool_feat, targets)
        return {'logits': b1_logits, 'feats': b1_pool_feat, 'targets': targets, 'camids': camids}
            # (b1_logits, b2_logits), \
            #    (b1_pool_feat, features, b2_pool_feat), \
            #        targets, camids

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        # images = [x["images"] for x in batched_inputs]
        """
        Normalize and batch the input images.
        """
        # images = [x["images"] for x in batched_inputs]
        images = batched_inputs["images"]
        images = (images - self.pixel_mean) / self.pixel_std
        images = images.detach()

        if self.training:
            #对一个批次里面的image，一半转换为另外一个模态
            def weighted(iamges):
                half = images[0::2]
                w = np.random.rand(3, 3)
                tmp = half.detach()
                for i in range(3):
                    t = w[i] / w[i].sum()
                    half[:, i] = t[0] * tmp[:, 0] + t[1] * tmp[:, 1] + t[2] * tmp[:, 2]

                images[0::2] = half
                return images

            def total(images):
                ori_images = images.detach()
                total = self._cfg.INPUT.CHANNEL.TOTAL
                if total.WEI_USE:
                    wei_images1 = 0.1 * images[:, 0] + 0.8 * images[:, 1] + 0.1 * images[:, 2]
                    wei_images1 = wei_images1.unsqueeze(1)

                    wei_images2 = 0.8 * images[:, 0] + 0.1 * images[:, 1] + 0.1 * images[:, 2]
                    wei_images2 = wei_images2.unsqueeze(1)

                    wei_images3 = 0.1 * images[:, 0] + 0.1 * images[:, 1] + 0.8 * images[:, 2]
                    wei_images3 = wei_images3.unsqueeze(1)

                    images = torch.cat((images, wei_images1, wei_images2, wei_images3), dim=1)
                if total.GRAY_USE:
                    gray_images = images[:, :3].mean(dim=1).unsqueeze(1)
                    images = torch.cat((images, gray_images), dim=1)

                M = images.shape[1]
                for i in range(len(images)):
                    ran_idx = np.random.choice(M)
                    images[i, 0] = images[i, ran_idx]
                    images[i, 1] = images[i, 0]
                    images[i, 2] = images[i, 1]

                images = images[:, :3].detach()

                if total.HALF_REPLACE:
                    for i in range(len(ori_images)):
                        ran_idxs = np.random.choice(3, 3)
                        ori_images[i, [0, 1, 2]] = ori_images[i, ran_idxs]

                    images[::2] = ori_images[::2]

                return images

            def shuffle(images):
                shuffle = self._cfg.INPUT.CHANNEL.SHUFFLE
                if shuffle.NAIVE:
                    for i in range(len(images)):
                        idxs = np.random.permutation(3)
                        images[i, [0,1,2]] = images[i, idxs]
                elif shuffle.HALF_REPLACE:
                    ran_idxs = np.random.choice(3, 3)
                    images[0::2, [0, 1, 2]] = images[0::2, ran_idxs]
                elif shuffle.TOTAL_REPLACE:
                    for i in range(len(images)):
                        ran_idxs = np.random.choice(3, 3)
                        images[i, [0, 1, 2]] = images[i, ran_idxs]
                elif shuffle.TOTAL_WEIGHTED:
                    for i in range(len(images)):
                        w = np.random.rand(3)
                        w = w / w.sum()
                        images[i, 0] = images[i, 0] * w[0] + images[i, 1] * w[1] + images[i, 2] * w[2]
                        images[i, 1] = images[i, 0]
                        images[i, 2] = images[i, 1]
                elif shuffle.ZERO_SEP:
                    for i in range(len(images)):
                        ran_idx = np.random.choice(3)
                        for j in range(3):
                            if j == ran_idx:
                                continue
                            images[i, j] = images[i, j] * 0.0
                            images[i, j] = images[i, j] * 0.0

                elif shuffle.TOTAL_SEP:
                    for i in range(len(images)):
                        ran_idx = np.random.choice(3)
                        images[i, 0] = images[i, ran_idx]
                        images[i, 1] = images[i, 0]
                        images[i, 2] = images[i, 1]
                elif shuffle.TOTAL_GRAY:
                    images = images.mean(dim=1).unsqueeze(1)
                    images = images.expand(-1, 3, -1, -1)
                else:
                    assert False, 'Muts be a True'

                return images

            def sep(images):
                images[:, [0, 1, 2]] = images[0:, np.random.choice(3, 3)]

                half = images[0::2]
                half = half[:, 0].unsqueeze(1)
                half = half.expand(-1, 3, -1, -1)
                images[0::2] = half
                return images

            def gray(images):
                half = images[0::2]
                half = half.mean(dim=1).unsqueeze(1)
                half = half.expand(-1, 3, -1, -1)
                images[0::2] = half

                return images

            if self._cfg.INPUT.CHANNEL.WEIGHTED:
                images = weighted(images)
            if self._cfg.INPUT.CHANNEL.SHUFFLE.ENABLED:
                images = shuffle(images)
            if self._cfg.INPUT.CHANNEL.SINGLE:
                images = sep(images)
            if self._cfg.INPUT.CHANNEL.GRAY:
                images = gray(images)
            if self._cfg.INPUT.CHANNEL.TOTAL.ENABLED:
                images = total(images)

            images.requires_grad = False
        return images


    def losses(self, outputs, use_repeatx3=False):
        logits, feats, targets, camids = outputs
        loss_dict = {}
        if use_repeatx3:
            loss_dict.update(reid_losses(self._cfg, logits[0][0::3], feats[0][0::3], targets[0::3], 'b1_ch1_'))
            loss_dict.update(reid_losses(self._cfg, logits[0][1::3], feats[0][1::3], targets[1::3], 'b1_ch2_'))
            loss_dict.update(reid_losses(self._cfg, logits[0][2::3], feats[0][2::3], targets[2::3], 'b1_ch3_'))
        else:
            loss_dict.update(reid_losses(self._cfg, logits[0], feats[0], targets, 'b1_'))

        return loss_dict