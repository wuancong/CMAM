#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import os
import logging
import sys

sys.path.append('.')
from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup
from fastreid.utils.checkpoint import Checkpointer
from fastreid.evaluation import ReidEvaluator
import time
import random
import torch
import numpy as np
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler


class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.grad_scaler = GradScaler()

    @classmethod
    def build_evaluator(cls, cfg, num_query):
        return ReidEvaluator(cfg, num_query)

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start
        """
        If your want to do something with the heads, you can wrap the model.
        """
        with autocast():
            outputs = self.model(data)
            loss_dict = self.model.losses(outputs)
            losses = sum(loss_item['value'] * loss_item['weight'] for loss_item in loss_dict.values())
            metrics_dict = {}
            for k, v in loss_dict.items():
                metrics_dict[k] = v['value']
            self._detect_anomaly(losses, metrics_dict)
            metrics_dict["data_time"] = data_time
            self._write_metrics(metrics_dict)
        """
        If you need accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        scaler = self.grad_scaler
        self.optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.unscale_(self.optimizer)
        scaler.step(self.optimizer)
        scaler.update()


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
    logger = logging.getLogger('fastreid.' + __name__)
    logger.info(f'seed = {seed}')

    if args.eval_only:
        model = Trainer.build_model(cfg)
        Checkpointer(model, save_dir=cfg.OUTPUT_DIR).load(cfg.MODEL.WEIGHTS)  # load trained model
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    gpus = args.gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    main(args)
