# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from ..registry import LOSSES
# from ..builder import build_backbone
@LOSSES.register_module()
class ContrastiveLoss(BaseModule):
    """Soft-Nearest Neighbors Loss for contrastive learning for multiple pos examples.

    The contrastive loss can be implemented in this head and is used in SimCLR,
    MoCo, DenseCL, etc.

    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Defaults to 0.1.
    the formula comes from this paper:
        "Analyzing and Improving Representations with the Soft Nearest Neighbor Loss"
    """
    def __init__(self, temperature=0.1, neck=None):
        super(ContrastiveLoss, self).__init__()
        # self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature
        # self.neck = build_backbone(neck) if neck else None 
    # def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False):
    #     super().__init__()
    #     if reduction not in ['none', 'mean', 'sum']:
    #         raise ValueError(f'Unsupported reduction mode: {reduction}. '
    #                          f'Supported ones are: {_reduction_modes}')

    #     self.loss_weight = loss_weight
    #     self.reduction = reduction
    #     self.sample_wise = sample_wise
    def forward(self, simMatrix, labels):
        """Forward function to compute contrastive loss.

        Args:
            simMatrix (Tensor): NxN matrix, all the similarities.
            labels (Tensor): Nx1 class label of each observation.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # print(labels)
        labels = labels.unsqueeze(1)
        labels_row = labels.permute(1,0)
        mask = torch.eq(labels_row,labels)
        lossall = []
        # from scipy.io import savemat

        # labelmat = labels.detach().cpu().numpy()
        # simMatrixmat = simMatrix.detach().cpu().numpy()
        # res = {'labels': labelmat, 'simMat': simMatrixmat}
        # savemat('/home/rui/Rui_SR/mmselfsup/matlab_labels_simMat.mat', res)
        # print('saved labels and simMatrix')
        # breakpoint()
        # print(simMatrix)
        # breakpoint()
        for idx, label in enumerate(labels):
            # choose the postive indices
            pos_tmp = (mask[idx]==True).nonzero(as_tuple=False)
            # print(pos_tmp,'111111111111')
            # breakpoint()
            # delete the sample itself
            pos_tmp = pos_tmp[pos_tmp!=idx].view(-1)
            single_sample = simMatrix[idx]
            """
            we already have the similarity matrix, no "-" needed, the paper use 
            L2 distance, so they have a "-" in the function
            """
            '''
            google research use exp(similarity/T) instead of exp(similarity)/T
            '''
            numerator = torch.sum(torch.exp(torch.index_select(single_sample, 0, pos_tmp)/self.temperature) )
            neg_tmp = torch.arange(0,len(labels))
            # delete the sample itself
            neg_tmp = neg_tmp[neg_tmp!=idx].to(numerator.device)
            denominator = torch.sum(torch.exp( torch.index_select(single_sample, 0, neg_tmp)/self.temperature ) ) + 1e-5
            # print(denominator,'2222222222222\n')
            single_loss = torch.log(numerator/denominator)
            lossall.append(single_loss)
            # print(numerator, denominator,single_loss)
        ######################################################
        # losses = dict()
        # losses['loss'] = -torch.mean(torch.stack(lossall))
        # return losses        
        #######################################################
        # breakpoint()
        # print(torch.stack(lossall))
        contrastive_loss = torch.stack(lossall)
        # breakpoint()
        # final_loss = contrastive_loss[torch.isfinite(contrastive_loss)]
        # print(contrastive_loss)
        return -torch.mean(contrastive_loss)
        '''
        tensor = torch.Tensor([[1, 2, 2, 7], [3, 1, 2, 4], [3, 1, 9, 4]])
        (tensor == 2).nonzero(as_tuple=False)
        '''
        # breakpoint()

    
    # def forward(self, pred, target, weight=None, **kwargs):
    #     """Forward Function.

    #     Args:
    #         pred (Tensor): of shape (N, C, H, W). Predicted tensor.
    #         target (Tensor): of shape (N, C, H, W). Ground truth tensor.
    #         weight (Tensor, optional): of shape (N, C, H, W). Element-wise
    #             weights. Default: None.
    #     """
    #     return self.loss_weight * l1_loss(
    #         pred,
    #         target,
    #         weight,
    #         reduction=self.reduction,
    #         sample_wise=self.sample_wise)