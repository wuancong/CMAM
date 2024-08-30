# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY
from .utils import *


@DATASET_REGISTRY.register()
class SYSUMM01(ImageDataset):
    def __init__(self, root='datasets', market1501_500k=False, **kwargs):
        # data = torch.load('/data/reid_dataset/sysu_mm01_re/dataset.pth')
        self.root = os.path.join(root, 'SYSU-MM01')
        train_ids = read_txt_and_convert_to_list(os.path.join(self.root, 'exp/train_id.txt'))
        val_ids = read_txt_and_convert_to_list(os.path.join(self.root, 'exp/val_id.txt'))
        train_val_ids = sorted(train_ids + val_ids)
        train_set = self.get_split_set(train_val_ids)
        test_ids = read_txt_and_convert_to_list(os.path.join(self.root, 'exp/test_id.txt'))
        test_set = self.get_split_set(test_ids)
        super(SYSUMM01, self).__init__(train_set, [], test_set, **kwargs)

    def get_split_set(self, train_ids):
        train_set = []
        for cam_ind in range(6):
            for train_ind, train_id in enumerate(train_ids):
                img_paths = find_jpg_files(os.path.join(self.root, f'cam{cam_ind + 1}', f'{train_id:04d}'))
                train_set += [(path, train_ind, cam_ind) for path in img_paths]
        return train_set
