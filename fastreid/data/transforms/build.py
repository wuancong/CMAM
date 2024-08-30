# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torchvision.transforms as T
from .transforms import *


def build_transforms(cfg, is_train=True):
    res = []

    if is_train:
        size_train = cfg.INPUT.SIZE_TRAIN

        # augmix augmentation
        do_augmix = cfg.INPUT.DO_AUGMIX

        # auto augmentation
        do_autoaug = cfg.INPUT.DO_AUTOAUG
        total_iter = cfg.SOLVER.MAX_ITER

        # horizontal filp
        do_flip = cfg.INPUT.DO_FLIP
        flip_prob = cfg.INPUT.FLIP_PROB

        # padding
        do_pad = cfg.INPUT.DO_PAD
        padding = cfg.INPUT.PADDING
        padding_mode = cfg.INPUT.PADDING_MODE

        # color jitter
        do_cj = cfg.INPUT.DO_CJ

        # random erasing
        do_rea = cfg.INPUT.REA.ENABLED
        do_crea = cfg.INPUT.REA.CREA
        rea_prob = cfg.INPUT.REA.PROB
        rea_mean = cfg.INPUT.REA.MEAN

        # random patch
        do_rpt = cfg.INPUT.RPT.ENABLED
        rpt_prob = cfg.INPUT.RPT.PROB

        res.append(T.Resize(size_train, interpolation=T.InterpolationMode.BICUBIC))
        if do_flip:
            res.append(T.RandomHorizontalFlip(p=flip_prob))
        if do_pad:
            res.extend([T.Pad(padding, padding_mode=padding_mode),
                        T.RandomCrop(size_train)])
        if do_cj:
            CJ = cfg.INPUT.CJ
            res.append(T.ColorJitter(brightness=CJ.BRI, contrast=CJ.CON, saturation=CJ.SAT, hue=CJ.HUE))
        if do_augmix:
            res.append(AugMix())
        if do_rea:
            if not do_crea:
                res.append(RandomErasing(probability=rea_prob, mean=rea_mean))
            else:
                res.append(CREA(probability=rea_prob, mean=rea_mean))
        if do_rpt:
            res.append(RandomPatch(prob_happen=rpt_prob))
    else:
        size_test = cfg.INPUT.SIZE_TEST
        res.append(T.Resize(size_test, interpolation=T.InterpolationMode.BICUBIC))

    res.append(ToTensor())
    return T.Compose(res)
