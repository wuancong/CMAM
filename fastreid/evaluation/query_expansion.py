# encoding: utf-8
"""
@author:  xingyu liao
@contact: liaoxingyu5@jd.com
"""

# based on
# https://github.com/PyRetri/PyRetri/blob/master/pyretri/index/re_ranker/re_ranker_impl/query_expansion.py

import numpy as np
import torch
import torch.nn.functional as F

def norm_dist(dists, isir):
    print("norm_dist")
    res = []
    for dis in dists:
        dis[isir==1] = (dis[isir==1] - dis[isir==1].mean()) / dis[isir==1].std()
        dis[isir==0] = (dis[isir==0] - dis[isir==0].mean()) / dis[isir==0].std()
        res.append(dis)
    res = torch.stack(res, 0)
    print(dis[isir==1].shape)
    return res

def aqe_aj(query_feat: torch.tensor, gallery_feat: torch.tensor,
        isir: torch.tensor, qe_times: int = 1, qe_k: int = 10, alpha: float = 3.0):
    """
    Combining the retrieved topk nearest neighbors with the original query and doing another retrieval.
    c.f. https://www.robots.ox.ac.uk/~vgg/publications/papers/chum07b.pdf
    Args :
        query_feat (torch.tensor):
        gallery_feat (torch.tensor):
        qe_times (int): number of query expansion times.
        qe_k (int): number of the neighbors to be combined.
        alpha (float):
    """
    num_query = query_feat.shape[0]
    all_feat = torch.cat((query_feat, gallery_feat), dim=0)
    norm_feat = F.normalize(all_feat, p=2, dim=1)

    all_feat = all_feat.numpy()
    for i in range(qe_times):
        all_feat_list = []
        sims = torch.mm(norm_feat, norm_feat.t())
        sims = norm_dist(sims, isir)
        sims = (sims - sims.min()) / (sims.max() - sims.min())
        sims = sims.data.cpu().numpy()
        for sim in sims:
            init_rank = np.argpartition(-sim, range(1, qe_k + 1))
            weights = sim[init_rank[:qe_k]].reshape((-1, 1))
            weights = np.power(weights, alpha)
            all_feat_list.append(np.mean(all_feat[init_rank[:qe_k], :] * weights, axis=0))
        all_feat = np.stack(all_feat_list, axis=0)
        norm_feat = F.normalize(torch.from_numpy(all_feat), p=2, dim=1)

    query_feat = torch.from_numpy(all_feat[:num_query])
    gallery_feat = torch.from_numpy(all_feat[num_query:])
    return query_feat, gallery_feat



def aqe(query_feat: torch.tensor, gallery_feat: torch.tensor,
        qe_times: int = 1, qe_k: int = 10, alpha: float = 3.0):
    """
    Combining the retrieved topk nearest neighbors with the original query and doing another retrieval.
    c.f. https://www.robots.ox.ac.uk/~vgg/publications/papers/chum07b.pdf
    Args :
        query_feat (torch.tensor):
        gallery_feat (torch.tensor):
        qe_times (int): number of query expansion times.
        qe_k (int): number of the neighbors to be combined.
        alpha (float):
    """
    num_query = query_feat.shape[0]
    all_feat = torch.cat((query_feat, gallery_feat), dim=0)
    norm_feat = F.normalize(all_feat, p=2, dim=1)

    all_feat = all_feat.numpy()
    for i in range(qe_times):
        all_feat_list = []
        sims = torch.mm(norm_feat, norm_feat.t())
        sims = sims.data.cpu().numpy()
        for sim in sims:
            init_rank = np.argpartition(-sim, range(1, qe_k + 1))
            weights = sim[init_rank[:qe_k]].reshape((-1, 1))
            weights = np.power(weights, alpha)
            all_feat_list.append(np.mean(all_feat[init_rank[:qe_k], :] * weights, axis=0))
        all_feat = np.stack(all_feat_list, axis=0)
        norm_feat = F.normalize(torch.from_numpy(all_feat), p=2, dim=1)

    query_feat = torch.from_numpy(all_feat[:num_query])
    gallery_feat = torch.from_numpy(all_feat[num_query:])
    return query_feat, gallery_feat

def pure_aqe(feat:torch.tensor, qe_times: int = 1, qe_k: int = 10, alpha: float = 3.0):
    query_feat = feat[:10]
    gallery_feat = feat[10:]
    query_feat, gallery_feat = aqe(query_feat, gallery_feat)
    return torch.cat((query_feat, gallery_feat), dim=0)

def queryexp(query_feat: torch.tensor, gallery_feat: torch.tensor,
        qe_times: int = 1, qe_k: int = 10, alpha: float = 3.0):
    """
    Combining the retrieved topk nearest neighbors with the original query and doing another retrieval.
    c.f. https://www.robots.ox.ac.uk/~vgg/publications/papers/chum07b.pdf
    Args :
        query_feat (torch.tensor):
        gallery_feat (torch.tensor):
        qe_times (int): number of query expansion times.
        qe_k (int): number of the neighbors to be combined.
        alpha (float):
    """
    num_query = query_feat.shape[0]
    # all_feat = torch.cat((query_feat, gallery_feat), dim=0)
    # norm_feat = F.normalize(all_feat, p=2, dim=1)
    norm_query = F.normalize(query_feat)
    norm_gallery = F.normalize(gallery_feat)

    query_feat = query_feat.numpy()
    gallery_feat = gallery_feat.numpy()

    all_feat_list = []
    sims = torch.mm(norm_query, norm_gallery.t())
    sims = sims.data.cpu().numpy()
    idx = 0
    for sim in sims:
        init_rank = np.argpartition(-sim, range(1, qe_k + 1))
        gidxs = init_rank[:qe_k]
        weights = sim[gidxs].reshape((-1, 1))
        weights = np.power(weights, alpha)
        gfeas = gallery_feat[gidxs, :] * weights
        qfea = query_feat[idx].reshape(1,-1)
        tfeats = np.concatenate((qfea, gfeas), 0 )
        all_feat_list.append(np.mean(tfeats, axis=0))
        idx = idx + 1
    all_feat = np.stack(all_feat_list, axis=0)

    all_feat = torch.from_numpy(all_feat)
    return all_feat

# def pure_qe(query_feat: torch.tensor, gallery_feat: torch.tensor,
#         qe_times: int = 1, qe_k: int = 10, alpha: float = 3.0):
#     gallery_np = gallery_feat.numpy()
#     query_np = query_feat.numpy()
#     norm_query = F.normalize(query_feat)
#     norm_gallery = F.normalize(gallery_feat)
#     sims = torch.mm(norm_query, norm_gallery)
#     sims = sims.data.cpu().numpy()
#     # query_np
#     for sim, qf in zip(sims, query_np):
#         init_rank = np.argpartition(-sim, range(1, qe_k + 1))
#         weights = sim[init_rank[:qe_k]].reshape((-1, 1))
#         weights = np.power(weights, alpha)
#         tfeat = np.mean(gallery_feat[init_rank[:qe_k], :] * weights)
#         tfeat = tfeat + fa