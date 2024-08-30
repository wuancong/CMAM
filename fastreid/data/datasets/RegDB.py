# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY
import os


@DATASET_REGISTRY.register()
class RegDB(ImageDataset):
    def __init__(self, root='datasets', **kwargs):
        for key, val in kwargs.items():
            if key == 'split_index':
                self.split_index = val
        train = self.process_dir('train')
        query = self.process_dir('query')
        gallery = self.process_dir('gallery')

        super(RegDB, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, mode):
        root = 'datasets/RegDB'
        assert os.path.isdir(root)
        assert mode in ['train', 'gallery', 'query']
        def loadIdx(index):
            Lines = index.readlines()
            idx = []
            for line in Lines:
                tmp = line.strip('\n')
                tmp = tmp.split(' ')
                idx.append(tmp)
            return idx

        num = str(self.split_index)
        if mode == 'train':
            index_RGB = loadIdx(open(root + '/idx/train_visible_'+num+'.txt','r'))
            index_IR  = loadIdx(open(root + '/idx/train_thermal_'+num+'.txt','r'))
        else:
            index_RGB = loadIdx(open(root + '/idx/test_visible_'+num+'.txt','r'))
            index_IR  = loadIdx(open(root + '/idx/test_thermal_'+num+'.txt','r'))

        if mode == 'gallery':
            img_paths = [root + '/' + path for path, _ in index_IR]
        elif mode == 'query':
            img_paths = [root + '/' + path for path, _ in index_RGB]
        else:
            img_paths = [root + '/' + path for path, _ in index_RGB] + [root + '/' + path for path, _ in index_IR]

        selected_ids = [int(path.split('/')[-2]) for path in img_paths]
        selected_ids = list(set(selected_ids))
        num_ids = len(selected_ids)

        img_paths = sorted(img_paths)
        self.img_paths = img_paths
        self.cam_ids = [int(path.split('/')[-3] == 'Thermal') for path in img_paths]
        self.num_ids = num_ids

        if mode == 'train':
            id_map = dict(zip(selected_ids, range(num_ids)))
            self.ids = [id_map[int(path.split('/')[-2])] for path in img_paths]
        else:
            self.ids = [int(path.split('/')[-2]) for path in img_paths]

        res = [(fname, pid, camid) for (fname, pid, camid) in zip(self.img_paths, self.ids, self.cam_ids)]

        return res