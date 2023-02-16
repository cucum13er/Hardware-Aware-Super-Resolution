# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseDataSource
from .cifar import CIFAR10, CIFAR100
from .image_list import ImageList
from .imagenet import ImageNet
################################################################
from .multidevice import MultiDevice
from .multidevice_ours import MultiDevice_ours
from .multidevice_ours_val import MultiDevice_ours_val
from .multidevice_ours_sepTarget import MultiDevice_ours_sepTarget
from .multidevice_ours_sepTarget_val import MultiDevice_ours_sepTarget_val
__all__ = ['BaseDataSource', 'CIFAR10', 'CIFAR100', 'ImageList', 'ImageNet', 'MultiDevice', 'MultiDevice_ours','MultiDevice_ours_val', 'MultiDevice_ours_sepTarget', 'MultiDevice_ours_sepTarget_val']
