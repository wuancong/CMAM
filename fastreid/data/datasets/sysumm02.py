import os.path as osp
from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY
import random

@DATASET_REGISTRY.register()
class SYSUMM02(ImageDataset):
    dataset_dir = 'SYSU-MM02'
    root = 'datasets'

    def __init__(self, root='datasets', **kwargs):
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.data_dir = self.dataset_dir
        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'test')
        self.gallery_dir = osp.join(self.data_dir, 'test')
        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)
        with open(osp.join(self.dataset_dir, 'train.txt'), 'r') as f:
            train_info = f.readlines()
            train_info = [e[:-1].split(' ') for e in train_info]
        with open(osp.join(self.dataset_dir, 'test_ir.txt'), 'r') as f:
            test_ir_info = f.readlines()
            test_ir_info = [e[:-1].split(' ') for e in test_ir_info]
        with open(osp.join(self.dataset_dir, 'test_rgb.txt'), 'r') as f:
            test_rgb_info = f.readlines()
            test_rgb_info = [e[:-1].split(' ') for e in test_rgb_info]

        train = [(osp.join(self.data_dir, e[0]), random.randint(0, 100), int(e[2]) - 1) for e in train_info]
        test_ir = [(osp.join(self.data_dir, e[0]), int(e[1]), int(e[2]) - 1) for e in test_ir_info]
        test_rgb = [(osp.join(self.data_dir, e[0]), int(e[1]), int(e[2]) - 1) for e in test_rgb_info]
        query = test_rgb
        gallery = test_ir
        super(SYSUMM02, self).__init__(train, query, gallery, **kwargs)
