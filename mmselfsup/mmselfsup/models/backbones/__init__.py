# Copyright (c) OpenMMLab. All rights reserved.
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .basic import BaseEncoder
###################################################################
from .easyres import EasyRes
__all__ = ['ResNet', 'ResNetV1d', 'ResNeXt', 'BaseEncoder', 'EasyRes']
