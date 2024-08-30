import torch.nn as nn
import math
import sys
import os.path as osp
import torch
from .build import BACKBONE_REGISTRY
import logging
from fastreid.utils.checkpoint import get_unexpected_parameters_message, get_missing_parameters_message
from .efficientnet_pytorch import EfficientNet
logger = logging.getLogger(__name__)

@BACKBONE_REGISTRY.register()
def build_effnet_backbone(cfg):
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

    if depth == 50:
        model = EfficientNet.from_pretrained('efficientnet-b4',weights_path=pretrain_path, advprop='adv' in pretrain_path)
    if depth == 100:
        model = EfficientNet.from_pretrained('efficientnet-b5', weights_path=pretrain_path,
                                             advprop='adv' in pretrain_path)
    return model