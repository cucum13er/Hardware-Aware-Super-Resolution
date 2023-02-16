# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import HEADS
from ..utils import accuracy


@HEADS.register_module()
class ClsHead_Twolayers(BaseModule):
    """Simplest classifier head, with only one fc layer.

    Args:
        with_avg_pool (bool): Whether to apply the average pooling
            after neck. Defaults to False.
        in_channels (int): Number of input channels. Defaults to 2048.
        num_classes (int): Number of classes. Defaults to 1000.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 with_avg_pool=False,
                 in_channels=2048,
                 hid_channels=128,
                 num_classes=5,
                 init_cfg=[
                     dict(type='Normal', std=0.01, layer='Linear'),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super(ClsHead_Twolayers, self).__init__(init_cfg)
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.num_classes = num_classes

        self.criterion = nn.CrossEntropyLoss()

        if self.with_avg_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc0 = nn.Linear(in_channels, hid_channels)
        self.fc_cls = nn.Linear(hid_channels, num_classes)

    def forward(self, x, labels):
        """Forward head.

        Args:
            x (list[Tensor] | tuple[Tensor]): Feature maps of backbone,
                each tensor has shape (N, C, H, W).

        Returns:
            list[Tensor]: A list of class scores.
        """
        assert isinstance(x, (tuple, list)) and len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            assert x.dim() == 4, \
                f'Tensor must has 4 dims, got: {x.dim()}'
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_feaures = self.fc0(x)
        # cls_feaures = nn.LeakyReLU(0.1)(cls_feaures)
        cls_score = self.fc_cls(cls_feaures)
        # from .snn_head import SNNLossHead
        # import torch
        # simMatrix = torch.matmul(cls_feaures, cls_feaures.permute(1, 0) )
        # loss_snn = SNNLossHead().forward(simMatrix,labels)
        # print(cls_score)
        # print(labels)
        # breakpoint()
        return [cls_score], cls_feaures

    def loss(self, cls_score, labels):
        """Compute the loss."""
        # print(cls_score[0])
        # print(labels)
        losses = dict()
        assert isinstance(cls_score, (tuple, list)) and len(cls_score) == 1
        losses['loss'] = self.criterion(cls_score[0], labels)
        losses['acc'] = accuracy(cls_score[0], labels)
        return losses
