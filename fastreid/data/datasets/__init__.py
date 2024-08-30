# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from ...utils.registry import Registry
DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets
It must returns an instance of :class:`Backbone`.
"""

from .market1501 import MARKET1501_TRANSFORMED_TO_MM01, MARKET1501_TRANSFORMED_TO_MM02, MARKET1501_TRANSFORMED_TO_REGDB
from .sysumm01 import SYSUMM01
from .sysumm02 import SYSUMM02
from .RegDB import RegDB

__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
