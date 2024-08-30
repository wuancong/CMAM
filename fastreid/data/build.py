# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import os
import torch
# from torch._six import container_abcs, string_classes, int_classes
import collections.abc as container_abcs
string_classes = str
int_classes = int
from torch.utils.data import DataLoader

from . import samplers
from .common import CommDataset
from .datasets import DATASET_REGISTRY
from .transforms import build_transforms
from .samplers.triplet_sampler import HardIdentitySampler

_root = os.getenv("FASTREID_DATASETS", "datasets")


def build_hardmining_loader(cfg,hard_samples,img_infos ):
    dataset = []
    batch_size = 2 * cfg.ADV.K + 1
    for sample in hard_samples:
        for j,dist in sample:
            dataset.append( img_infos[j] )
    test_transforms = build_transforms(cfg, is_train=False)
    dataset = CommDataset(dataset, test_transforms, relabel=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_sampler = samplers.InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, batch_size, False)
    test_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=fast_batch_collator)
    return test_loader


def build_hardmining_train_loader(cfg, dataset, train_set, datasetinfos):
    num_workers = cfg.DATALOADER.NUM_WORKERS
    batch_size = cfg.SOLVER.IMS_PER_BATCH
    num_instance = cfg.DATALOADER.NUM_INSTANCE
    data_sampler = HardIdentitySampler(dataset, datasetinfos, num_instance)
    batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, batch_size, True)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=fast_batch_collator,
    )
    return train_loader

def build_reid_traverse_loader(cfg, dataset):
    if not isinstance(dataset,CommDataset):
        test_transforms = build_transforms(cfg, is_train=False)
        dataset = CommDataset(dataset, test_transforms, relabel=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    batch_size = cfg.TEST.IMS_PER_BATCH
    data_sampler = samplers.InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, batch_size, False)
    test_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=fast_batch_collator)
    return test_loader

def build_reid_ultrain_loader(cfg, dataset, sampler='BASE'):
    train_transforms = build_transforms(cfg, is_train=True)
    train_set = CommDataset(dataset, train_transforms, relabel=True, cfg=cfg)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    batch_size = cfg.SOLVER.IMS_PER_BATCH
    num_instance = cfg.DATALOADER.NUM_INSTANCE

    data_sampler = samplers.BalancedIdentitySampler(train_set.img_items, batch_size, num_instance)
    batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, batch_size, True)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=fast_batch_collator,
        pin_memory=True,
    )
    return train_loader

def build_reid_train_loader(cfg): # the train loader built here is not used for unsupervised learning
    train_transforms = build_transforms(cfg, is_train=True)

    train_items = list()
    max_id = 0
    for d in cfg.DATASETS.NAMES:
        if d == 'RegDB':
            dataset = DATASET_REGISTRY.get(d)(root=_root, combineall=cfg.DATASETS.COMBINEALL, split_index=cfg.DATASETS.REGDB_SPLIT_INDEX)
        else:
            dataset = DATASET_REGISTRY.get(d)(root=_root, combineall=cfg.DATASETS.COMBINEALL)
        dataset.show_train()
        dataset.train = [(fn, pid + max_id, cid) for (fn, pid, cid) in dataset.train]
        train_items.extend(dataset.train)
        max_id = max([pid for _, pid, _ in train_items]) + 1

    train_set = CommDataset(train_items, train_transforms, relabel=True, cfg=cfg)
    if cfg.MODEL.HEADS.NUM_CLASSES == -1:
        cfg.defrost()
        cfg.MODEL.HEADS.NUM_CLASSES = len(train_set.pid_dict)
        cfg.freeze()
    num_workers = cfg.DATALOADER.NUM_WORKERS
    batch_size = cfg.SOLVER.IMS_PER_BATCH
    num_instance = cfg.DATALOADER.NUM_INSTANCE

    data_sampler = samplers.BalancedIdentitySampler(train_set.img_items, batch_size, num_instance)
    batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, batch_size, True)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=fast_batch_collator,
        pin_memory=True
    )
    return train_loader


def build_reid_test_loader(cfg, dataset_name):
    test_transforms = build_transforms(cfg, is_train=False)

    if dataset_name == 'RegDB':
        dataset = DATASET_REGISTRY.get(dataset_name)(root=_root, split_index=cfg.DATASETS.REGDB_SPLIT_INDEX)
    else:
        dataset = DATASET_REGISTRY.get(dataset_name)(root=_root)
    dataset.show_test()
    test_items = dataset.query + dataset.gallery

    test_set = CommDataset(test_items, test_transforms, relabel=False, cfg=cfg)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    batch_size = cfg.TEST.IMS_PER_BATCH
    data_sampler = samplers.InferenceSampler(len(test_set))
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, batch_size, False)
    test_loader = DataLoader(
        test_set,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=fast_batch_collator)
    return test_loader, len(dataset.query)


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch

def fast_batch_collator(batched_inputs):
    """
    A simple batch collator for most common reid tasks
    """
    elem = batched_inputs[0]
    if isinstance(elem, torch.Tensor):
        out = torch.zeros((len(batched_inputs), *elem.size()), dtype=elem.dtype)
        for i, tensor in enumerate(batched_inputs):
            out[i] += tensor
        return out

    elif isinstance(elem, container_abcs.Mapping):
        return {key: fast_batch_collator([d[key] for d in batched_inputs]) for key in elem}

    elif isinstance(elem, float):
        return torch.tensor(batched_inputs, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batched_inputs)
    elif isinstance(elem, string_classes):
        return batched_inputs
