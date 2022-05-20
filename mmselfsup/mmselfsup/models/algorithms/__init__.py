# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseModel
from .byol import BYOL
from .classification import Classification
from .deepcluster import DeepCluster
from .densecl import DenseCL
from .moco import MoCo
from .npid import NPID
from .odc import ODC
from .relative_loc import RelativeLoc
from .rotation_pred import RotationPred
from .simclr import SimCLR
from .simsiam import SimSiam
from .swav import SwAV
#################################################
from .simclr_multidevice import SimCLR_Multidevice
from .simclr_multidevice_cls import SimCLR_Multidevice_cls
from .simclr_multidevice_nolabel import SimCLR_Nolabel
from .moco_label import MoCo_label
__all__ = [
    'BaseModel', 'BYOL', 'Classification', 'DeepCluster', 'DenseCL', 'MoCo',
    'NPID', 'ODC', 'RelativeLoc', 'RotationPred', 'SimCLR', 'SimSiam', 'SwAV', 'SimCLR_Multidevice', 'SimCLR_Multidevice_cls', 'SimCLR_Nolabel',
    'MoCo_label'
]
