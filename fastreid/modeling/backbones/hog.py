# encoding: utf-8
import torch
from torch import nn
from cv2 import HOGDescriptor
import numpy as np

class HOGExtractor(nn.Module):
    def __init__(self, win_size=(128, 256), block_size=(32, 32), block_stride=(16, 16), cell_size=(16, 16), nbins=9):
        super(HOGExtractor, self).__init__()
        self.hog = HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

    def forward(self, x):
        imgs = x['images']
        imgs = imgs.numpy().astype(np.uint8).transpose((0, 2, 3, 1))
        feat_list = []
        for img in imgs:
            feat = self.hog.compute(img)
            feat = np.pad(feat, (0, 4096-3780), 'constant', constant_values=(0.0, 0.0))
            feat_list.append(torch.from_numpy(feat))
        return (torch.stack(feat_list, dim=0), x['targets'], x['camid'], x['img_path'])
