
import mmedit.models.backbones.sr_backbones.common as common
import torch.nn.functional as F
#from option import args as args_global
import torch
from mmcv.runner import load_checkpoint
from torch import nn
from mmedit.models.builder import build_backbone
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
from mmedit.models.backbones.sr_backbones.dasr import DASR

mynet = DASR(3,3)


###############################################################################
#test lq and gt if they are matched
import matplotlib.pyplot as plt
import torch
lq = torch.load('/home/rui/Rui_SR/mmediting/work_dirs/restorers/dasr/X2/transfer_ours_frozen0_crop96/lq.pt')
gt = torch.load('/home/rui/Rui_SR/mmediting/work_dirs/restorers/dasr/X2/transfer_ours_frozen0_crop96/gt.pt')

plt.imshow(lq[3].permute(1, 2, 0))
plt.imshow(gt[3].permute(1, 2, 0))
# class DASR(nn.Module):
#     def __init__(   self,
#                     in_channels,
#                     out_channels,
#                     kernel_size = 3,
#                     mid_channels = 64,
#                     num_blocks = 5,
#                     num_groups = 5,
#                     upscale_factor = 4,
#                     num_layers = 8,
#                     scale = 4,
#                     reduction = 8,
#                     frozen_groups = 0,
#                     # deg_rand = False,
#                 ):