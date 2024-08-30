#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import os
import logging
import sys
import os.path as osp

sys.path.append('.')
from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup
from fastreid.utils.checkpoint import Checkpointer
from fastreid.data.build import build_reid_traverse_loader, build_reid_ultrain_loader
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.evaluation import (ReidEvaluator, inference_on_dataset)
from fastreid.utils.faiss_rerank import compute_jaccard_distance
import time
import random
import torch
import numpy as np
from fast_pytorch_kmeans import KMeans
from fastreid.layers.cm import ClusterMemory
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import v_measure_score
from sklearn.cluster import DBSCAN
import torch.nn.functional as F
import copy
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


def merge_clusters(centroids, labels, k):
    z = linkage(pdist(centroids), method='ward')  # 使用Ward方法计算簇间距离
    label_map_old = fcluster(z, k, criterion='maxclust')
    label_map = copy.deepcopy(label_map_old)
    for i, l in enumerate(np.unique(label_map_old)):
        label_map[label_map_old == l] = i
    new_labels = []
    for l in labels:
        if l == -1:
            new_labels.append(l)
        else:
            new_labels.append(label_map[l])
    new_labels = torch.tensor(new_labels)
    return new_labels


def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):

    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.

    original_dist = np.concatenate(
      [np.concatenate([q_q_dist, q_g_dist], axis=1),
       np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
      axis=0)
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2.))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2.))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)
    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)


    for i in range(query_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist


def pairwise_distance(x, y):
    x, y = x.cuda(), y.cuda()
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    x = F.normalize(x)
    y = F.normalize(y)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist = dist.addmm(x, y.t(), beta=1, alpha=-2)
    return dist.cpu()


class Trainer(DefaultTrainer):

    def __init__(self, cfg):
        self.memorys = {}
        self.previous_step = 0
        self.data_length = 0
        self.cluster_method_idx = 0
        super(Trainer, self).__init__(cfg)

    @classmethod
    def build_evaluator(cls, cfg, num_query, output_folder=None):
        return ReidEvaluator(cfg, num_query)

    def build_something(self, cfg):
        from torch.cuda.amp import GradScaler
        self.grad_scaler = GradScaler()
        self.i_train = 0
        if cfg.DATASETS.NAMES[0] == 'SYSUMM01':
            self.rgb_cams, self.ir_cams = [0, 1, 3, 4], [2, 5]
        elif cfg.DATASETS.NAMES[0] == 'SYSUMM02':
            self.rgb_cams, self.ir_cams = [0], [1, 2]
        elif cfg.DATASETS.NAMES[0] == 'RegDB':
            self.rgb_cams, self.ir_cams = [0], [1]
        else:
            raise ValueError('Unsupported cfg.DATASETS.NAMES')

    def get_d_feature(self, cfg, dataset, input_key='images', use_model_idx=[1, 2]):
        data_loader = build_reid_traverse_loader(cfg, dataset)
        evaluator = ReidEvaluator(cfg, 0)
        inference_on_dataset(self.model, data_loader, evaluator, flip_test=cfg.TEST.FLIP, not_eval=True, input_key=input_key, use_model_idx=use_model_idx)
        features = torch.cat(evaluator.features, dim=0)
        img_paths = np.array(evaluator.fns)

        return img_paths, features, dataset

    def get_feature(self, dataset_name, input_key='images', use_model_idx=[1, 2]):
        cfg = self.cfg
        if dataset_name == 'RegDB':
            dataset = DATASET_REGISTRY.get(dataset_name)(root="datasets", combineall=cfg.DATASETS.COMBINEALL, split_index=cfg.DATASETS.REGDB_SPLIT_INDEX)
        else:
            dataset = DATASET_REGISTRY.get(dataset_name)(root="datasets", combineall=cfg.DATASETS.COMBINEALL)
        return self.get_d_feature(cfg, dataset.train, input_key=input_key, use_model_idx=use_model_idx)

    @staticmethod
    def compute_centers(features, labels):
        centers = []
        labels_re = labels
        num_clusters = labels_re.max() + 1
        for i in range(num_clusters):
            if (labels_re == i).sum() > 0:
                center = features[labels_re == i].mean(dim=0)
            else:
                center = features[0] * 0.0
            # center = center / torch.norm(center)
            centers.append(center)
        centers = torch.stack(centers, 0)
        return centers

    def make_memory(self, center_dict):
        for k, cluster_centers in center_dict.items():
            memory = ClusterMemory(self.cfg, cluster_centers.size(1), cluster_centers.size(0), temp=self.cfg.UL.CLUSTER.MEMORY_TEMP).cuda()
            memory.features = F.normalize(cluster_centers, dim=1).cuda()
            self.memorys[k] = memory

    def save_train_feature(self, cfg):
        data_name = cfg.DATASETS.NAMES[0]
        img_paths, features, trainset = self.get_feature(data_name)
        pids = [pid for fn, pid, cid in trainset]
        fns = [fn for fn, pid, cid in trainset]
        camids = [cid for fn, pid, cid in trainset]
        save_content = {'features': features, 'pids': pids, 'camids': camids,
                        'fns': fns}
        save_path = osp.join(cfg.OUTPUT_DIR, f'{data_name}_features.pth')
        torch.save(save_content, save_path)

    def cluster(self):
        cfg = self.cfg
        CLUSTER = cfg.UL.CLUSTER
        data_name = cfg.DATASETS.NAMES[0]
        _, features_all, trainset_all = self.get_feature(data_name, use_model_idx=[1, 2])

        gt_all = np.array([pid for fn, pid, cid in trainset_all])
        sup_id_num = 0

        gt = np.array(gt_all)
        features = features_all
        trainset = trainset_all

        isir = [camid in self.ir_cams for fname, pid, camid in trainset]
        isir = np.array(isir)
        irgt = gt[isir]
        rgbgt = gt[~isir]
        isir = torch.tensor(isir)
        logger = logging.getLogger('fastreid.' + __name__)

        ir_feas = features[isir]
        rgb_feas = features[isir == 0]
        ir_concat_feas = ir_feas
        rgb_concat_feas = rgb_feas

        def pseudo_label_assignment(features, method_name, num_clusters=0, eps=0.0):
            if method_name == 'DBSCAN':
                rerank_dist = compute_jaccard_distance(features.cuda(), k1=args.k1, k2=args.k2, use_float16=False)
                cluster_method = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=None)
                pseudo_labels = cluster_method.fit_predict(rerank_dist)
                num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
                return torch.from_numpy(pseudo_labels), num_ids
            elif method_name == 'KMEANS':
                cluster_method = KMeans(n_clusters=num_clusters, mode='cosine', verbose=1)
                pseudo_labels = cluster_method.fit_predict(features.cuda())
                return pseudo_labels.cpu(), num_clusters
            else:
                raise(ValueError('unsuported cluster method'))

        if CLUSTER.METHOD == 'CROSS_DBSCAN_BGM':
            cluster_name = 'DBSCAN'
            logger.info(f'{cluster_name} CLUSTERING')

            rgb_cluster_feas = rgb_concat_feas
            ir_cluster_feas = ir_concat_feas

            rgbidxs, rgb_num_cluster = pseudo_label_assignment(rgb_cluster_feas, cluster_name, num_clusters=CLUSTER.NUM, eps=args.eps)
            v = v_measure_score(rgbgt, rgbidxs.cpu().numpy())
            logger.info('rgb v_measure:{}'.format(np.round(v, 3)))
            logger.info('rgb cluster num:{}'.format(np.round(rgb_num_cluster, 3)))

            iridxs, ir_num_cluster = pseudo_label_assignment(ir_cluster_feas, cluster_name, num_clusters=CLUSTER.NUM, eps=args.eps)

            v = v_measure_score(irgt, iridxs.cpu().numpy())
            logger.info('ir v_measure:{}'.format(np.round(v, 3)))
            logger.info('ir cluster num:{}'.format(np.round(ir_num_cluster, 3)))

            self.cfg.defrost()
            self.cfg.UL.CLUSTER.NUM = ir_num_cluster
            self.cfg.freeze()

            labels_intra = torch.zeros(len(isir), dtype=torch.long)
            rgbidxs_intra = copy.deepcopy(rgbidxs)
            iridxs_intra = copy.deepcopy(iridxs)
            labels_intra[isir] = iridxs_intra
            labels_intra[isir == 0] = rgbidxs_intra

            rgb_is_valid, ir_is_valid = rgbidxs != -1, iridxs != -1

            ircens_cluster = self.compute_centers(ir_cluster_feas, iridxs)
            rgbcens_cluster = self.compute_centers(rgb_cluster_feas, rgbidxs)

            if len(rgbcens_cluster) > len(ircens_cluster):
                rgbidxs = merge_clusters(centroids=rgbcens_cluster, labels=rgbidxs, k=len(ircens_cluster))

            elif len(rgbcens_cluster) < len(ircens_cluster):
                iridxs = merge_clusters(centroids=ircens_cluster, labels=iridxs, k=len(rgbcens_cluster))

            rgbcens = self.compute_centers(rgb_concat_feas, rgbidxs)
            ircens = self.compute_centers(ir_concat_feas, iridxs)

            dist = pairwise_distance(rgbcens, ircens)

            rows, cols = linear_sum_assignment(dist.cpu().numpy())
            idx = np.argsort(rows)
            cols = cols[idx]
            rgb2ir = torch.from_numpy(cols).cuda()

            iridxs = iridxs.cpu()
            rgbidxs_valid = rgb2ir[rgbidxs[rgb_is_valid]].cpu()
            rgbidxs[rgb_is_valid] = rgbidxs_valid

            rgbidxs_selected = rgbidxs
            iridxs_selected = iridxs

            v = v_measure_score(rgbgt[rgbidxs_selected != -1], rgbidxs_selected[rgbidxs_selected != -1])
            logger.info('rgb selected v_measure:{}'.format(np.round(v, 3)))
            v = v_measure_score(irgt[iridxs_selected != -1], iridxs_selected[iridxs_selected != -1])
            logger.info('ir selected v_measure:{}'.format(np.round(v, 3)))
            labels = torch.zeros(len(isir), dtype=torch.long)
            labels[isir] = iridxs_selected.type(torch.long)
            labels[isir == 0] = rgbidxs_selected.type(torch.long)
            v = v_measure_score(gt[labels != -1], labels[labels != -1].numpy())
            logger.info('total v_measure: {}'.format(np.round(v, 3)))
            logger.info('total sample num: {}'.format(len(gt[labels != -1])))

        elif CLUSTER.METHOD == 'CROSS_KMEANS_BGM':
            num_clusters = CLUSTER.NUM - sup_id_num
            logger.info('KMEANS CLUSTERING') # map label from ir to rgb
            rgbidxs, rgb_num_cluster = pseudo_label_assignment(rgb_concat_feas, 'KMEANS', num_clusters=num_clusters)
            v = v_measure_score(rgbgt, rgbidxs.cpu().numpy())
            logger.info('rgb v_measure:{}'.format(np.round(v, 3)))
            logger.info('rgb cluster num:{}'.format(np.round(rgb_num_cluster, 3)))

            iridxs, ir_num_cluster = pseudo_label_assignment(ir_concat_feas, 'KMEANS', num_clusters=num_clusters)
            v = v_measure_score(irgt, iridxs.cpu().numpy())
            logger.info('ir v_measure:{}'.format(np.round(v, 3)))
            logger.info('ir cluster num:{}'.format(np.round(ir_num_cluster, 3)))

            labels_intra = torch.zeros(len(isir), dtype=torch.long)
            labels_intra[isir == 0] = rgbidxs
            labels_intra[isir] = iridxs

            rgbcens = self.compute_centers(rgb_concat_feas, rgbidxs)
            ircens = self.compute_centers(ir_concat_feas, iridxs)
            dist = pairwise_distance(rgbcens, ircens)

            rows, cols = linear_sum_assignment(dist.cpu().numpy())
            idx = np.argsort(rows)
            cols = cols[idx]
            rgb2ir = torch.from_numpy(cols).cuda()

            labels = torch.zeros(len(isir), dtype=torch.long)
            labels[isir] = iridxs.cpu()
            labels[isir == 0] = rgb2ir[rgbidxs].cpu()

            v = v_measure_score(gt, labels)
            logger.info('total v_measure:{}'.format(np.round(v, 3)))
        else:
            raise ValueError('Unsupported clustering method.')

        # compute memory centers
        features_m1, features_m2 = torch.chunk(features, chunks=2, dim=1)
        centers1, centers2 = self.compute_centers(features_m1, labels), self.compute_centers(features_m2, labels)
        self.make_memory({'share': centers1, 'share2': centers2})

        new_data = []
        for (fname, _, camid), label in zip(trainset, labels):
            label = label.item()
            if label == -1:
                continue
            tmp = (fname, label, camid)
            new_data.append(tmp)

        logger.info(f'training sample num: {len(new_data)}')
        loader = build_reid_ultrain_loader(cfg, new_data, sampler=CLUSTER.SAMPLER)
        self._data_loader_iter = iter(loader)
        self.data_length = len(new_data)
        fn = osp.join(cfg.OUTPUT_DIR, cfg.SAVE_PRE + '_pseudolabel.pth')
        torch.save(new_data, fn)

    def run_step(self):
        cfg = self.cfg

        if self.cfg.UL.CLUSTER.TIMES < 0: # cluster after training one epoch
            if (self.i_train - self.previous_step + 1) * self.cfg.SOLVER.IMS_PER_BATCH > self.data_length:
                print(f'cluster at step {self.i_train}')
                self.cluster()
                self.previous_step = self.i_train
        else: # cluster after UL.CLUSTER.TIMES times of iteration
            if self.i_train % self.cfg.UL.CLUSTER.TIMES == 0:
                self.cluster()

        from torch.cuda.amp import autocast
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """
        with autocast():
            data = next(self._data_loader_iter)
            data_time = time.perf_counter() - start
            """
            If your want to do something with the heads, you can wrap the model.
            """
            outputs = self.model(data)
            loss_dict = self.model.uda_losses(outputs, self.memorys)
            losses = sum(loss_item['value'] * loss_item['weight'] for loss_item in loss_dict.values())

            metrics_dict = {}
            for k, v in loss_dict.items():
                metrics_dict[k] = v['value']
            self._detect_anomaly(losses, metrics_dict)

            metrics_dict["data_time"] = data_time
            self._write_metrics(metrics_dict)

        scaler = self.grad_scaler
        self.optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.unscale_(self.optimizer)
        scaler.step(self.optimizer)
        scaler.update()

        self.i_train += 1

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    seed = args.seed
    if seed == -1:
        seed = int(time.time())
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    cfg = setup(args)

    if args.save_train_feature: # cache train feature
        trainer = Trainer(cfg)
        trainer.build_something(cfg)
        trainer.resume_or_load(resume=args.resume)
        trainer.save_train_feature(cfg)
        return

    logger = logging.getLogger('fastreid.' + __name__)
    logger.info(f'seed = {seed}')
    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = Trainer.build_model(cfg)
        Checkpointer(model, save_dir=cfg.OUTPUT_DIR).load(cfg.MODEL.WEIGHTS)  # load trained model
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.build_something(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()
    logger.info(f'cluster_num={trainer.cfg.UL.CLUSTER.NUM}')


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    gpus = args.gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    main(args)
