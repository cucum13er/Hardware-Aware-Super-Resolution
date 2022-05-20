# Copyright (c) OpenMMLab. All rights reserved.
import torch
from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from ..utils import GatherLayer
from .base import BaseModel


@ALGORITHMS.register_module()
class SimCLR_Multidevice(BaseModel):
    """SimCLR.

    Implementation of `A Simple Framework for Contrastive Learning
    of Visual Representations <https://arxiv.org/abs/2002.05709>`_.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact feature
            vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self, backbone, neck=None, head=None, init_cfg=None):
        super(SimCLR_Multidevice, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        assert head is not None
        self.head = build_head(head)

    @staticmethod
    def _create_buffer(N):
        """Compute the mask and the index of positive samples.

        Args:
            N (int): batch size.
        """
        mask = 1 - torch.eye(N * 2, dtype=torch.uint8).cuda()
        pos_ind = (torch.arange(N * 2).cuda(),
                   2 * torch.arange(N, dtype=torch.long).unsqueeze(1).repeat(
                       1, 2).view(-1, 1).squeeze().cuda())
        neg_mask = torch.ones((N * 2, N * 2 - 1), dtype=torch.uint8).cuda()
        neg_mask[pos_ind] = 0
        return mask, pos_ind, neg_mask

    def extract_feat(self, img):
        """Function to extract features from backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone(img)
        # x = self.neck(x)
        return x
    
    def forward_test(self, img, label, **kwargs):
        with torch.no_grad():
            
            assert isinstance(img, list)
            num_views = len(img)
            label_all = label.repeat(num_views)
            # print(label_all)
            img = torch.cat(img)        
            x = self.extract_feat(img)  # 2n 128*2048*7*7 N C H W
            # cls_score, _ =  self.head(x, label_all) # two layers classification
            # losses = self.head.loss(cls_score, label_all)

 
            x = self.neck(x)  # (2n)xd 128*128 N C
            # breakpoint()
            results = dict()
            results.update(zip(x[0].cpu(), label_all.cpu())) 
        return results
        
    def forward_train(self, img, label, **kwargs):#####################add label
        """Forward computation during training.

        Args:
            img (list[Tensor]): A list of input images with shape
                (N, C, H, W). Typically these should be mean centered
                and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert isinstance(img, list)
        """
        img = num_views * batch_size
        img[0] is first view, img[1] is second view
        label = batch_size
        e.g., batch size = 64, views = 2
        img = list of 2, each list contains 64 images
        label = list of 64, represents the gt_label of each image
        """
        # get all the labels for each image
        # print(img[0].shape)
        num_views = len(img)
        label_all = label.repeat(num_views)
        # print(label_all)
        ###prove the img[0] and img[1] are the same image on the same index
        # import matplotlib.pyplot as plt
        # imgplot = plt.imshow(img[0][0,:,:,:].to('cpu').permute(1,2,0))
        # plt.show()
        # imgplot = plt.imshow(img[1][0,:,:,:].to('cpu').permute(1,2,0))
        # plt.show()
        # print(label_all.shape,'11111111\n')
        # breakpoint()
        # exit()        
        # cat all images together
        img = torch.cat(img)
        # import matplotlib.pyplot as plt
        # imgplot = plt.imshow(img[0,:,:,:].to('cpu').permute(1,2,0))
        # plt.show()
        # imgplot = plt.imshow(img[2,:,:,:].to('cpu').permute(1,2,0))
        # plt.show()
        # breakpoint()
        # exit()           
        x = self.extract_feat(img)  # 2n 128*2048*7*7 N C H W
        # print(x[0].shape, '33333333333333333\n')        
        z = self.neck(x)[0]  # (2n)xd 128*128 N C
        z = z / (torch.norm(z, p=2, dim=1, keepdim=True) + 1e-10)
        z = torch.cat(GatherLayer.apply(z), dim=0)  # (2N)xd
 
        # assert z.size(0) % 2 == 0
        # N = z.size(0) // 2 # 64 images (batch size)
        s = torch.matmul(z, z.permute(1, 0))  # (2N)x(2N)
        # print(s)
        # print(label_all,'1111111111111111\n')
        # breakpoint()
        #######################################################################
        # mask, pos_ind, neg_mask = self._create_buffer(N)
        
        # # remove diagonal, (2N)x(2N-1)
        # s = torch.masked_select(s, mask == 1).reshape(s.size(0), -1)
        # positive = s[pos_ind].unsqueeze(1)  # (2N)x1
        # # print(pos_ind,'33333333333333\n')
        # # breakpoint()
        # # exit() 
        # # select negative, (2N)x(2N-2)
        # negative = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)
        #######################################################################
        
        losses = self.head(s,label_all) # new SNN Loss head
        # losses = self.head(positive, negative)
        # print(losses)
        return losses
"""
pos_ind:

(tensor([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
         14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
         28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
         42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
         56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
         70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
         84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
         98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
        112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
        126, 127], device='cuda:0'), 
 tensor([  0,   0,   2,   2,   4,   4,   6,   6,   8,   8,  10,  10,  12,  12,
         14,  14,  16,  16,  18,  18,  20,  20,  22,  22,  24,  24,  26,  26,
         28,  28,  30,  30,  32,  32,  34,  34,  36,  36,  38,  38,  40,  40,
         42,  42,  44,  44,  46,  46,  48,  48,  50,  50,  52,  52,  54,  54,
         56,  56,  58,  58,  60,  60,  62,  62,  64,  64,  66,  66,  68,  68,
         70,  70,  72,  72,  74,  74,  76,  76,  78,  78,  80,  80,  82,  82,
         84,  84,  86,  86,  88,  88,  90,  90,  92,  92,  94,  94,  96,  96,
         98,  98, 100, 100, 102, 102, 104, 104, 106, 106, 108, 108, 110, 110,
        112, 112, 114, 114, 116, 116, 118, 118, 120, 120, 122, 122, 124, 124,
        126, 126], device='cuda:0')) 

"""
# [[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3],[4,4,4,4,4],[5,5,5,5,5]]

