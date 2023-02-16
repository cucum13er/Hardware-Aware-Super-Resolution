# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import ALGORITHMS, build_backbone, build_head
from ..utils import Sobel
from .base import BaseModel
from mmcv.runner import BaseModule, auto_fp16
from ..heads.cls_head_twolayers import ClsHead_Twolayers
@ALGORITHMS.register_module()
class Classification(BaseModel):
    """Simple image classification.

    Args:
        backbone (dict): Config dict for module of backbone.
        with_sobel (bool): Whether to apply a Sobel filter.
            Defaults to False.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self, backbone, with_sobel=False, head=None, init_cfg=None):
        super(Classification, self).__init__(init_cfg)
        self.with_sobel = with_sobel
        if with_sobel:
            self.sobel_layer = Sobel()
        self.backbone = build_backbone(backbone)
        assert head is not None
        self.head = build_head(head)
    
    
    def extract_feat(self, img):
        """Function to extract features from backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        if self.with_sobel:
            img = self.sobel_layer(img)
        x = self.backbone(img)
        return x

    def forward_train(self, img, label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            labels (Tensor): Ground-truth labels.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # print(labels,'1111111111111111111\n')
        x = self.extract_feat(img)
        # breakpoint()
        # print(x[0].shape, '1111111111111111111111\n')
        num_views = x[0].size(0) // label.size(0)
        labels = label.repeat(num_views)
        # print(labels,'2222222222222222222222\n')
        if isinstance(self.head, ClsHead_Twolayers):
            cls_score, _ =  self.head(x, labels)
            losses = self.head.loss(cls_score, labels)

            
        else:
            outs = self.head(x, labels)##########################
            loss_inputs = (outs, labels)
            losses = self.head.loss(*loss_inputs)
            # print(labels,'111111111111111111\n')
            # breakpoint()
            # exit()
        return losses

    def forward_test(self, img, label, **kwargs):
        """Forward computation during test.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of output features.
        """
        x = self.extract_feat(img)  # tuple
        num_views = x[0].size(0) // label.size(0)
        labels = label.repeat(num_views)
        cls_score, _ =  self.head(x, labels)
        
        # breakpoint()
        # save the img of the test data
        # from torchvision.utils import save_image
        # save_image(img, 'work_dirs/selfsup/moco/transfer_moco_easyres_Ours_supcon_S1/')
        # losses = self.head.loss(cls_score, labels)
        # outs = self.head(x)
        # keys = [f'head{i}' for i in self.backbone.out_indices]
        # out_tensors = [out.cpu() for out in outs]  # NxC
        # print(cls_score)
        # print(labels)
        # breakpoint()
        return dict(zip(cls_score[0].cpu(), labels.cpu()))
