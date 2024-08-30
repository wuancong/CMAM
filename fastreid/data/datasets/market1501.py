# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY
import torch


def process_dir(dir_path):
    img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
    img_paths = sorted(img_paths)
    pattern = re.compile(r'([-\d]+)_c(\d)')
    data = []
    for img_path in img_paths:
        pid, camid = map(int, pattern.search(img_path).groups())
        if pid == -1:
            continue  # junk images are just ignored
        assert 0 <= pid <= 1501  # pid == 0 means background
        assert 1 <= camid <= 6
        camid -= 1  # index starts from 0
        data.append((img_path, pid, camid))
    return data


class MARKET1501_TRANSFORMED(ImageDataset):
    def __init__(self,
                 data_gen_train, #default sampling steps for diffusion model is 20
                 data_gen_test,
                 **kwargs):
        self.root = 'datasets'
        self.dataset_dir = osp.join(self.root, 'market1501_gen')
        data = process_dir(osp.join(self.dataset_dir, 'market1501_train')) + \
               process_dir(osp.join(self.dataset_dir, 'market1501_test'))
        data_gen = process_dir(osp.join(self.dataset_dir, data_gen_train)) + \
                   process_dir(osp.join(self.dataset_dir, data_gen_test))
        data_pseudo = torch.load(osp.join(self.dataset_dir, 'pplr_pseudo_labels.pth'))
        fname_pid_dict = {}
        for d in data_pseudo:
            fname_pid_dict[osp.basename(d[0])] = d[1]
        train = []
        for d1, d2 in zip(data, data_gen):
            d1_basename = osp.basename(d1[0])
            assert d1_basename == osp.basename(d2[0])
            pid = fname_pid_dict.get(d1_basename)
            if pid is not None:
                train.append(((d1[0], d2[0]), pid, d1[2]))
        query = train
        gallery = train
        super(MARKET1501_TRANSFORMED, self).__init__(train, query, gallery, **kwargs)


@DATASET_REGISTRY.register()
class MARKET1501_TRANSFORMED_TO_MM01(MARKET1501_TRANSFORMED):
    def __init__(self,
                 data_gen_train='market1501_train_sysumm01',
                 data_gen_test='market1501_test_sysumm01',
                 **kwargs):
        super(MARKET1501_TRANSFORMED_TO_MM01, self).__init__(data_gen_train, data_gen_test, **kwargs)


@DATASET_REGISTRY.register()
class MARKET1501_TRANSFORMED_TO_MM02(MARKET1501_TRANSFORMED):
    def __init__(self,
                 data_gen_train='market1501_train_sysumm02',
                 data_gen_test='market1501_test_sysumm02',
                 **kwargs):
        super(MARKET1501_TRANSFORMED_TO_MM02, self).__init__(data_gen_train, data_gen_test, **kwargs)

@DATASET_REGISTRY.register()
class MARKET1501_TRANSFORMED_TO_REGDB(MARKET1501_TRANSFORMED):
    def __init__(self,
                 data_gen_train='market1501_train_regdb',
                 data_gen_test='market1501_test_regdb',
                 **kwargs):
        super(MARKET1501_TRANSFORMED_TO_REGDB, self).__init__(data_gen_train, data_gen_test, **kwargs)
