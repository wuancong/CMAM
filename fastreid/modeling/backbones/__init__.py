# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from .build import build_backbone, BACKBONE_REGISTRY

from .resnet import build_resnet_backbone
from .osnet import build_osnet_backbone
from .resnest import build_resnest_backbone
from .resnext import build_resnext_backbone
from .BIT import build_bit_backbone
from .effnet import build_effnet_backbone
from .swi_resnet import build_swi_resnet_backbone
from .snr_resnet import build_snr_resnet_backbone
from .SWI import build_swi_backbone
from .SWI_ATT import build_swi_att_backbone
from .swi_adaresnet import build_swi_adaresnet_backbone
from .swi_resnet_adno import build_swi_resnet_addno_backbone
from .swi_resnet_adno2 import build_swi_resnet_addno2_backbone
from .dra_resnet import build_kdr_swi_resnet_backbone
from .base_mpa import build_resnetmpa_backbone
from .hog import HOGExtractor
