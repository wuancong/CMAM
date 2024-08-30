# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import random
from collections import defaultdict
import copy

import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import BatchSampler

def No_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]

class IRContrastSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // self.num_instances
        rgb_cams = [0, 1, 3, 4]
        ir_cams = [2, 5]
        self.camids = [cid for _, _, cid in self.data_source]
        N = len(self.camids)
        iridxs = [idx for idx in range(N) if self.camids[idx] in ir_cams]
        rgbidxs = [idx for idx in range(N) if self.camids[idx] not in ir_cams]

        self.iridxs = iridxs
        self.rgbidxs = rgbidxs
        self._seed = 0
    def __iter__(self):
        np.random.seed(self._seed)

        while True:
            final_idxs = []
            iridxs = np.random.choice(self.iridxs, size=self.batch_size//4, replace=True)
            rgbidxs = np.random.choice(self.rgbidxs, size=self.batch_size//4, replace=True)

            for idx in iridxs:
                final_idxs.append(idx)
                final_idxs.append(idx)

            for idx in rgbidxs:
                final_idxs.append(idx)
                final_idxs.append(idx)


            yield from final_idxs

class UDABalancedBatchSampler_ir_rgb(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples, dataset_name):
        if dataset_name == 'sysu-mm01':
            rgb_cams = [0, 1, 3, 4]
            ir_cams = [2, 5]
        elif dataset_name == 'sysu-mm02':
            rgb_cams = [2]
            ir_cams = [0, 1]
        elif dataset_name == 'regdb':
            rgb_cams = [1]
            ir_cams = [2]
        else:
            raise Exception('dataset not found in sampler')
        rgb_indices = []
        ir_indices = []
        rgb_labels = []
        ir_labels = []
        for i in range(len(dataset)):
            if dataset[i][2] in rgb_cams:
                rgb_labels.append(dataset[i][1])
                rgb_indices.append(i)
            elif dataset[i][2] in ir_cams:
                ir_labels.append(dataset[i][1])
                ir_indices.append(i)
            else:
                raise Exception('cams filter error')

        rgb_indices = np.asarray(rgb_indices)
        ir_indices = np.asarray(ir_indices)
        rgb_labels = np.asarray(rgb_labels)
        ir_labels = np.asarray(ir_labels)

        self.labels = np.zeros(len(dataset),).astype(int)
        self.data_length = max(len(ir_indices), len(rgb_indices))*2

        for i in range(len(dataset)):
            self.labels[i] = dataset[i][1]
        self.labels_set = list(set(self.labels))
        print('label_set', self.labels_set)
        self.rgb_label_to_indices = {label: rgb_indices[np.where(rgb_labels==label)] for label in self.labels_set}
        self.ir_label_to_indices = {label: ir_indices[np.where(ir_labels == label)] for label in self.labels_set}

        #test
        count = 0
        for k in self.rgb_label_to_indices.keys():
            count += len(self.rgb_label_to_indices[k])
        print ('rgb count ', count, len(rgb_indices))
        count = 0
        for k in self.ir_label_to_indices.keys():
            count += len(self.ir_label_to_indices[k])
        print('ir count ', count, len(ir_indices))


        for l in self.labels_set:
            np.random.shuffle(self.rgb_label_to_indices[l])
            np.random.shuffle(self.ir_label_to_indices[l])
        #add the record the used indices in each label
        self.used_rgb_label_indices_count = {label: 0 for label in self.labels_set}
        self.used_ir_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes
        print ('initial done')

    def __iter__(self):
        self.count = 0
        while True:
            ret = []
            classes = np.random.choice(self.labels_set, int(self.n_classes), replace=False)
            indices = []
            for class_ in classes:
                #choose n_sample/2 rgb and n_sample/2 ir
                rgb_idxs = self.rgb_label_to_indices[class_]
                ir_idxs = self.ir_label_to_indices[class_]
                lrgb = len(rgb_idxs)
                lir = len(ir_idxs)
                if lrgb > 0 and lir > 0:
                    i1 = np.random.choice(rgb_idxs, self.n_samples // 2, replace=lrgb<self.n_samples//2)
                    i2 = np.random.choice(ir_idxs, self.n_samples // 2, replace=lir<self.n_samples//2)
                    indices = i1.tolist() + i2.tolist()
                elif lrgb > 0:
                    i1 = np.random.choice(rgb_idxs, self.n_samples, replace=lrgb<self.n_samples)
                    indices = i1.tolist()
                else:
                    i2 = np.random.choice(ir_idxs, self.n_samples, replace=lir<self.n_samples)
                    indices = i2.tolist()

            ret.extend(indices)

            yield from ret

    def __len__(self):
        return (self.data_length // self.batch_size)*self.batch_size
class BalancedBatchSampler_ir_rgb(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples, dataset_name):
        if dataset_name == 'sysu-mm01':
            rgb_cams = [0, 1, 3, 4]
            ir_cams = [2, 5, 6]
        elif dataset_name == 'sysu-mm02':
            rgb_cams = [2]
            ir_cams = [0, 1]
        elif dataset_name == 'regdb':
            rgb_cams = [1]
            ir_cams = [2]
        else:
            raise Exception('dataset not found in sampler')
        rgb_indices = []
        ir_indices = []
        rgb_labels = []
        ir_labels = []
        for i in range(len(dataset)):
            if dataset[i][2] in rgb_cams:
                rgb_labels.append(dataset[i][1])
                rgb_indices.append(i)
            elif dataset[i][2] in ir_cams:
                ir_labels.append(dataset[i][1])
                ir_indices.append(i)
            else:
                raise Exception('cams filter error')

        rgb_indices = np.asarray(rgb_indices)
        ir_indices = np.asarray(ir_indices)
        rgb_labels = np.asarray(rgb_labels)
        ir_labels = np.asarray(ir_labels)

        self.labels = np.zeros(len(dataset),).astype(int)
        self.data_length = max(len(ir_indices), len(rgb_indices))*2

        for i in range(len(dataset)):
            self.labels[i] = dataset[i][1]
        self.labels_set = list(set(self.labels))
        print('label_set', self.labels_set)
        self.rgb_label_to_indices = {label: rgb_indices[np.where(rgb_labels==label)] for label in self.labels_set}
        self.ir_label_to_indices = {label: ir_indices[np.where(ir_labels == label)] for label in self.labels_set}

        #test
        count = 0
        for k in self.rgb_label_to_indices.keys():
            count += len(self.rgb_label_to_indices[k])
        print ('rgb count ', count, len(rgb_indices))
        count = 0
        for k in self.ir_label_to_indices.keys():
            count += len(self.ir_label_to_indices[k])
        print('ir count ', count, len(ir_indices))


        for l in self.labels_set:
            np.random.shuffle(self.rgb_label_to_indices[l])
            np.random.shuffle(self.ir_label_to_indices[l])
        #add the record the used indices in each label
        self.used_rgb_label_indices_count = {label: 0 for label in self.labels_set}
        self.used_ir_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes
        print ('initial done')

    def __iter__(self):
        self.count = 0
        while True:
            ret = []
            classes = np.random.choice(self.labels_set, int(self.n_classes), replace=False)
            indices = []
            for class_ in classes:
                #choose n_sample/2 rgb and n_sample/2 ir

                indices.extend(self.rgb_label_to_indices[class_][
                               self.used_rgb_label_indices_count[class_]:self.used_rgb_label_indices_count[
                                                                         class_] + self.n_samples//2])
                self.used_rgb_label_indices_count[class_] += self.n_samples//2
                if self.used_rgb_label_indices_count[class_] + self.n_samples//2 > len(self.rgb_label_to_indices[class_]):
                    np.random.shuffle(self.rgb_label_to_indices[class_])
                    self.used_rgb_label_indices_count[class_] = 0

                indices.extend(self.ir_label_to_indices[class_][
                               self.used_ir_label_indices_count[class_]:self.used_ir_label_indices_count[
                                                                             class_] + self.n_samples // 2])
                self.used_ir_label_indices_count[class_] += self.n_samples // 2
                if self.used_ir_label_indices_count[class_] + self.n_samples // 2 > len(self.ir_label_to_indices[class_]):
                    np.random.shuffle(self.ir_label_to_indices[class_])
                    self.used_ir_label_indices_count[class_] = 0


            ret.extend(indices)

            yield from ret

    def __len__(self):
        return (self.data_length // self.batch_size)*self.batch_size


class HardIdentitySampler(Sampler):
    def __init__(self, data_source, data_infos, num_instances=16):
        self.data_infos = data_infos
        self.num_instances = num_instances
        self.num_identities = len(data_infos)
        self._seed = 0
        self._shuffle = True
        self.num_pids_per_batch = 4
        self.pids = [pid for _, pid, _ in data_source]
        self.pids = np.array(self.pids)
    def __iter__(self):
        indices = self._infinite_indices()
        for kid in indices:
            ret = []
            ir_idxs, rgb_idxs, topkidxs = self.data_infos[kid]
            sel_iridxs = np.random.choice(len(ir_idxs), size=self.num_instances//2, replace=True)
            sel_idxs = ir_idxs[sel_iridxs]
            sel_rgb_idxs =  topkidxs[sel_iridxs]
            sel_rgb_idxs = rgb_idxs[sel_rgb_idxs]
            ret = sel_idxs.tolist() + sel_rgb_idxs.tolist()

            yield from ret

    def _infinite_indices(self):
        np.random.seed(self._seed)
        while True:
            if self._shuffle:
                identities = np.random.permutation(self.num_identities)
            else:
                identities = np.arange(self.num_identities)
            drop_indices = self.num_identities % self.num_pids_per_batch
            if drop_indices == 0:
                yield from identities
            yield from identities[:-drop_indices]


class BalancedIdentitySampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances=4):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // self.num_instances

        self.index_pid = defaultdict(list)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)

        for index, info in enumerate(data_source):
            pid = info[1]
            camid = info[2]
            self.index_pid[index] = pid
            self.pid_cam[pid].append(camid)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_identities = len(self.pids)

        self._seed = 0
        self._shuffle = True

    def __iter__(self):
        indices = self._infinite_indices()
        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])
            _, i_pid, i_cam = self.data_source[i][0], self.data_source[i][1], self.data_source[i][2]
            ret = [i]
            pid_i = self.index_pid[i]
            cams = self.pid_cam[pid_i]
            index = self.pid_index[pid_i]
            select_cams = No_index(cams, i_cam)

            if select_cams: # select samples from different cameras of sample i
                if len(select_cams) >= self.num_instances:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=False)
                else:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=True)
                for kk in cam_indexes:
                    ret.append(index[kk])
            else:
                select_indexes = No_index(index, i)
                if not select_indexes:
                    # only one image for this identity
                    ind_indexes = [0] * (self.num_instances - 1)
                elif len(select_indexes) >= self.num_instances:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=False)
                else:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=True)

                for kk in ind_indexes:
                    ret.append(index[kk])
            yield from ret

    def _infinite_indices(self):
        np.random.seed(self._seed)
        while True:
            if self._shuffle:
                # print(self.num_identities)
                identities = np.random.permutation(self.num_identities)
            else:
                identities = np.arange(self.num_identities)
            drop_indices = self.num_identities % self.num_pids_per_batch
            if drop_indices == 0:
                yield from identities
            yield from identities[:-drop_indices]


class NaiveIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // self.num_instances

        self.index_pid = defaultdict(list)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)

        for index, info in enumerate(data_source):
            pid = info[1]
            camid = info[2]
            self.index_pid[index] = pid
            self.pid_cam[pid].append(camid)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_identities = len(self.pids)

        self._seed = 0

    def __iter__(self):
        np.random.seed(self._seed)

        while True:
            batch_idxs_dict = defaultdict(list)

            for pid in self.pids:
                idxs = copy.deepcopy(self.pid_index[pid])
                if len(idxs) < self.num_instances:
                    idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
                random.shuffle(idxs)
                batch_idxs = []
                for idx in idxs:
                    batch_idxs.append(idx)
                    if len(batch_idxs) == self.num_instances:
                        batch_idxs_dict[pid].append(batch_idxs)
                        batch_idxs = []

            avai_pids = copy.deepcopy(self.pids)
            final_idxs = []

            while len(avai_pids) >= self.num_pids_per_batch:
                selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
                for pid in selected_pids:
                    batch_idxs = batch_idxs_dict[pid].pop(0)
                    final_idxs.extend(batch_idxs)
                    if len(batch_idxs_dict[pid]) == 0:
                        avai_pids.remove(pid)
            yield from final_idxs


class BalancedModalitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, rgb_ir_cams):
        self.rgb_cams, self.ir_cams = rgb_ir_cams
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_rgb_cams_per_batch = batch_size // (2 * len(self.rgb_cams))
        self.num_ir_cams_per_batch = batch_size // (2 * len(self.ir_cams))

        self.index_rgb_cam = defaultdict(list)
        self.rgb_cam_index = defaultdict(list)
        self.index_ir_cam = defaultdict(list)
        self.ir_cam_index = defaultdict(list)
        self.rgb_index = []
        self.ir_index = []

        for index, info in enumerate(data_source):
            cam = info[2]
            if cam in self.rgb_cams:
                self.index_rgb_cam[index] = cam
                self.rgb_cam_index[cam].append(index)
                self.rgb_index.append(index)
            elif cam in self.ir_cams:
                self.index_ir_cam[index] = cam
                self.ir_cam_index[cam].append(index)
                self.ir_index.append(index)
        self.data_length = max(len(self.ir_index), len(self.rgb_index)) * 2
        self._seed = 0

    def __iter__(self):
        np.random.seed(self._seed)
        while True:
            iridxs = np.random.choice(self.ir_index, size=self.batch_size // 2, replace=True)
            rgbidxs = np.random.choice(self.rgb_index, size=self.batch_size // 2, replace=True)
            final_idxs = list(iridxs) + list(rgbidxs)
            yield from final_idxs

    def __len__(self):
        return (self.data_length // self.batch_size)*self.batch_size
