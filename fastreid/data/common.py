# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from .data_utils import read_image
import copy
import torchvision.transforms as T
from PIL import Image


class HPAUG():
    def __init__(self, cfg):
        HP = cfg.INPUT.HP
        bri, con, sat, hue = HP.BRI, HP.CON, HP.SAT, HP.HUE
        self.taug = T.ColorJitter(brightness=bri, contrast=con, saturation=sat, hue=hue)
        if HP.PTHFN:
            self.img2parse = torch.load(HP.PTHFN)

        self.NOISE = HP.NOISE
        self.use_hp = HP.ENABLED
        self.prob = HP.PROB

    def __call__(self, img_fn, img):
        if not self.use_hp:
            return img

        if img_fn not in self.img2parse:
            return img

        sta_img = self.taug(img)
        parse_fn = self.img2parse[img_fn]
        parse = Image.open(parse_fn)

        img = np.asarray(img)
        parse = np.asarray(parse)
        sta_img = np.asarray(sta_img) + (np.random.rand(*img.shape) * self.NOISE - (self.NOISE // 2)).astype(np.uint8)

        W, H = parse.shape
        mask = (parse.reshape(W, H, 1) == 0).astype(np.uint8)
        img = img * (1 - mask) + sta_img * mask
        return Image.fromarray(img)


class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True, cfg=None):
        self.transform = transform
        self.relabel = relabel
        self.pid_dict = {}
        self.cfg = cfg
        self.img_items = img_items
        if isinstance(self.img_items[0][0], tuple): # read multiple images for one sample, including original image and transformed image)
            self.num_img_per_sample = len(self.img_items[0][0])
        else:
            self.num_img_per_sample = 1
        if self.relabel:
            self.pids = set([item[1] for item in img_items])
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path_this, pid, camid = self.img_items[index]
        if self.relabel:
            pid = self.pid_dict[pid]
        if self.num_img_per_sample == 1:
            img_path = img_path_this
        else:
            img_path = img_path_this[0]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return_dict = {
            'images': img,
            'targets': pid,
            'camid': camid,
            'img_path': img_path,
        }

        if self.num_img_per_sample == 2:
            img_i = read_image(img_path_this[1])
            if self.transform is not None:
                img_i = self.transform(img_i)
            return_dict['images1'] = img_i
        return return_dict


    def update_pid_dict(self, pid_dict):
        self.pid_dict = pid_dict

