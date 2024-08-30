# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from .evaluator import DatasetEvaluator
from .evaluate_SYSU_MM01 import evaluate_results as evaluate_mm01
from .eval_regdb import eval_regdb
from fastreid.data.datasets.utils import read_txt_and_convert_to_list

logger = logging.getLogger(__name__)


def get_absame(a, b):
    M = len(a)
    N = len(b)
    a = np.tile(a.reshape(M, 1), (1, N))
    b = np.tile(b.reshape(1, N), (M, 1))

    return a == b


class ReidEvaluator(DatasetEvaluator):
    best_result = [0.0]
    def __init__(self, cfg, num_query, output_dir=None):
        self.cfg = cfg
        self._num_query = num_query
        self._output_dir = output_dir

        self.features = []
        self.pids = []
        self.camids = []

    def reset(self):
        self.features = []
        self.pids = []
        self.camids = []
        self.fns = []

    def process(self, outputs):
        self.features.append(outputs[0].cpu())
        self.pids.extend(outputs[1].cpu().numpy())
        self.camids.extend(outputs[2].cpu().numpy())
        self.fns.extend(outputs[3])

    @staticmethod
    def cal_dist(metric: str, query_feat: torch.tensor, gallery_feat: torch.tensor):
        assert metric in ["cosine", "euclidean"], "must choose from [cosine, euclidean], but got {}".format(metric)
        if metric == "cosine":
            query_feat = F.normalize(query_feat, dim=1)
            gallery_feat = F.normalize(gallery_feat, dim=1)
            dist = 1 - torch.mm(query_feat, gallery_feat.t())
        else:
            m, n = query_feat.size(0), gallery_feat.size(0)
            xx = torch.pow(query_feat, 2).sum(1, keepdim=True).expand(m, n)
            yy = torch.pow(gallery_feat, 2).sum(1, keepdim=True).expand(n, m).t()
            dist = xx + yy
            # dist.addmm_(1, -2, query_feat, gallery_feat.t())
            dist.addmm_(query_feat, gallery_feat.t(), beta=1, alpha=-2)
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist.cpu().numpy()

    def hard_mininig(self, K):
        #For large data, we only need first _num_query to adv.
        features = torch.cat(self.features[:self._num_query], dim=0)
        pids = np.asarray(self.pids[:self._num_query])
        camids = np.asarray( self.camids[:self._num_query] )
        dist = self.cal_dist(self.cfg.TEST.METRIC, features,features)
        # print( 'dist.shape', dist.shape )
        pidsame = get_absame(pids,pids)
        camsame = get_absame(camids,camids)

        neg_list = []
        #Same Camera but not same person
        neg_dist = copy.deepcopy(dist)
        neg_dist[~camsame] = np.inf
        neg_dist[pidsame] = np.inf
        index_array = np.argpartition(neg_dist, kth=K, axis=-1)
        for i, x in enumerate(index_array[:, :K]):
            tmp = []
            for j in x:
                tmp.append( (j,neg_dist[i,j]) )
            neg_list.append( tmp )

        #Cross Camera but same person
        pos_list = []
        pos_dist = copy.deepcopy(dist)
        pos_dist[camsame] = np.inf
        pos_dist[~pidsame] = np.inf
        index_array = np.argpartition(pos_dist,kth=K,axis = -1 )
        for i, x in enumerate(index_array[:, :K]):
            tmp = []
            for j in x:
                if np.isinf( pos_dist[i,j] ):
                    continue
                tmp.append( (j, pos_dist[i,j]) )
            if len(tmp) == 0 or len(tmp) == K:
                pos_list.append( tmp )
            else:
                ranidxs = np.random.choice(len(tmp), K - len(tmp))
                for idx in ranidxs:
                    tmp.append( tmp[idx] )
                pos_list.append( tmp )
        res = []
        for i, (pos, neg) in enumerate( zip(pos_list,neg_list) ):
            tmp = []
            if len(pos) == 0:
                continue
            tmp.append( (i,0.0) )
            tmp.extend( pos )
            tmp.extend( neg )

            res.append( tmp )

        return res

    def evaluate(self):
        features = torch.cat(self.features, dim=0)
        query_features = features[:self._num_query]
        query_pids = np.asarray(self.pids[:self._num_query])
        query_camids = np.asarray(self.camids[:self._num_query])

        gallery_features = features[self._num_query:]
        gallery_pids = np.asarray(self.pids[self._num_query:])
        gallery_camids = np.asarray(self.camids[self._num_query:])

        if self.cfg.DATASETS.TESTS[0] in ['SYSUMM02', 'RegDB']:
            q_ids, q_cams = query_pids, query_camids
            g_ids, g_cams = gallery_pids, gallery_camids

            q_feats = query_features[:, :2048]
            g_feats = gallery_features[:, :2048]
            logger.info('model1 results')
            logger.info('query rgb, gallery ir')
            cmc, mAP = eval_regdb(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, logger=logger)
            logger.info(f'Rank-1, Rank-5, Rank-10: {cmc[0]}, {cmc[4]}, {cmc[9]}')
            logger.info(f'mAP: {mAP}')
            logger.info('query ir, gallery rgb')
            cmc, mAP = eval_regdb(g_feats, g_ids, g_cams, q_feats, q_ids, q_cams, logger=logger)
            logger.info(f'Rank-1, Rank-5, Rank-10: {cmc[0]}, {cmc[4]}, {cmc[9]}')
            logger.info(f'mAP: {mAP}')

            result = cmc

            # q_feats = query_features[:, 2048:]
            # g_feats = gallery_features[:, 2048:]
            # logger.info('model2 results')
            # logger.info('query rgb, gallery ir')
            # cmc, mAP = eval_regdb(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, logger=logger)
            # logger.info(f'Rank-1, Rank-5, Rank-10: {cmc[0]}, {cmc[4]}, {cmc[9]}')
            # logger.info(f'mAP: {mAP}')
            # logger.info('query ir, gallery rgb')
            # cmc, mAP = eval_regdb(g_feats, g_ids, g_cams, q_feats, q_ids, q_cams, logger=logger)
            # logger.info(f'Rank-1, Rank-5, Rank-10: {cmc[0]}, {cmc[4]}, {cmc[9]}')
            # logger.info(f'mAP: {mAP}')

            if ReidEvaluator.best_result[0] < result[0]:
                ReidEvaluator.best_result = result
            logger.info('Best rank-1 result: ' + str(ReidEvaluator.best_result[0]))
        else:
            #SYSU-MM01
            npfeatures = features.numpy()
            test_ids = read_txt_and_convert_to_list('datasets/SYSU-MM01/exp/test_id.txt')
            features_dict_model1 = {}
            features_dict_model2 = {}
            features_dict_concat = {}
            for cam_ind in range(6):
                features_dict_model1[f'cam{cam_ind+1}'] = [np.array([]) for i in range(333)]
                features_dict_model2[f'cam{cam_ind+1}'] = [np.array([]) for i in range(333)]
                features_dict_concat[f'cam{cam_ind+1}'] = [np.array([]) for i in range(333)]
                for pid_ind, test_id in enumerate(test_ids):
                    features_model1 = [npfeatures[i][:2048] for i in range(len(features)) if self.pids[i] == pid_ind and self.camids[i] == cam_ind]
                    features_model2 = [npfeatures[i][2048:] for i in range(len(features)) if self.pids[i] == pid_ind and self.camids[i] == cam_ind]
                    features_concat = [npfeatures[i] for i in range(len(features)) if self.pids[i] == pid_ind and self.camids[i] == cam_ind]
                    if features_model1:
                        features_model1 = np.vstack(features_model1)
                        features_model2 = np.vstack(features_model2)
                        features_concat = np.vstack(features_concat)
                    else:
                        continue
                    features_dict_model1[f'cam{cam_ind+1}'][test_id-1] = features_model1
                    features_dict_model2[f'cam{cam_ind+1}'][test_id-1] = features_model2
                    features_dict_concat[f'cam{cam_ind+1}'][test_id-1] = features_concat

            mode = 'all'
            cmc, mAP = evaluate_mm01(features_dict_model1, np.array(test_ids) - 1, mode=mode)
            result = cmc
            logger.info(f'Model 1 {mode} results')
            logger.info(f'Rank-1, Rank-5, Rank-10: {cmc[0]}, {cmc[4]}, {cmc[9]}')
            logger.info(f'mAP: {mAP}')
            mode = 'indoor'
            cmc, mAP = evaluate_mm01(features_dict_model1, np.array(test_ids) - 1, mode=mode)
            logger.info(f'Model 1 {mode} results')
            logger.info(f'Rank-1, Rank-5, Rank-10: {cmc[0]}, {cmc[4]}, {cmc[9]}')
            logger.info(f'mAP: {mAP}')
            # cmc, mAP = evaluate_mm01(features_dict_model2, np.array(test_ids) - 1)
            # logger.info('Model 2 results')
            # logger.info(f'Rank-1, Rank-5, Rank-10: {cmc[0]}, {cmc[4]}, {cmc[9]}')
            # logger.info(f'mAP: {mAP}')
            # cmc, map = evaluate_mm01(features_dict_concat, np.array(test_ids) - 1)
            # logger.info('Model 1 + model 2 results')
            # logger.info(f'Rank-1, Rank-5, Rank-10: {cmc[0]}, {cmc[4]}, {cmc[9]}')
            # logger.info(f'mAP: {map}')
            if ReidEvaluator.best_result[0] < result[0]:
                ReidEvaluator.best_result = result
            logger.info('Best rank-1 result: ' + str(ReidEvaluator.best_result[0]))
        self._results = OrderedDict()
        return copy.deepcopy(self._results)

