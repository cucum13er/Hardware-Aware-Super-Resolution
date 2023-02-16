
import mmedit.models.backbones.sr_backbones.common as common
import torch.nn.functional as F
#from option import args as args_global
import torch
from mmcv.runner import load_checkpoint
from torch import nn
from mmedit.models.builder import build_backbone
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
class SA_layer(nn.Module):
    def __init__(self):
        super(SA_layer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(512, 8*8)
    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        b, c, h, w = x[0].size()
        feature = self.avgpool(x[1]) # b*512*1*1
        feature = feature.view(feature.size(0), -1) # b*512
        # breakpoint()
        # self.projection = nn.Linear(feature.size(1), h*w).to('cuda')
        # print(feature.size(1),h,w)
        feature = self.projection(feature)
        # split the feature map into 16 by 16 regions
        
        feature = feature.view(b,1,8,8)
        # assert h%16 == 0
        # assert w%16 == 0
        if h%8==0 and w%8==0:
            feature = torch.repeat_interleave(feature, h//8, dim=2)
            feature = torch.repeat_interleave(feature, w//8, dim=3)
        else:
            print('warning: spatial attention value is not well aligned (8*8)! Use iterpolation instead!')    
            feature = nn.functional.interpolate(feature, size=(h,w), mode='bicubic' )
        out = x[0] * feature
        # breakpoint()
        return out 
    
class DA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, reduction, deg_fsize=512):
        super(DA_conv, self).__init__()
        # self.channels_out = channels_out
        # self.channels_in = channels_in
        # self.kernel_size = kernel_size
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.adaptive_size = nn.Linear(deg_fsize, self.channels_in, bias=False)
        #revise the kernel into adaptive to input parameters 
        # self.kernel = nn.Sequential(
        #     nn.Linear(self.channels_in, self.channels_in//reduction, bias=False),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(self.channels_in//reduction, self.channels_out * self.kernel_size * self.kernel_size, bias=False)
        # )
        # self.conv = nn.Conv2d(channels_in, channels_out, 1, padding=1//2, bias=True)
        # self.conv = common.default_conv(channels_in, channels_out, 1)
        self.sa = SA_layer()
        self.ca = CA_layer(channels_in, channels_out, reduction)
        
        # self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        # print(x[0].size())
        # b, c, h, w = x[0].size()
        
        # # breakpoint()
        # # branch 1
        # feature = self.avgpool(x[1]) # b*2048*1*1
        # feature = feature.view(feature.size(0), -1) # b*2048
        # # self.adaptive_size = nn.Linear(feature.size(1), self.channels_in, bias=False,device='cuda')
        # # breakpoint()
        # feature = self.adaptive_size(feature)
        # kernel = self.kernel(feature).view(-1, 1, self.kernel_size, self.kernel_size)
        # out = self.relu(F.conv2d(x[0].view(1, -1, h, w), kernel, groups=b*c, padding=(self.kernel_size-1)//2))
        # out = self.conv(out.view(b, -1, h, w))
        # # breakpoint()
        # # branch 2
        out = self.sa(x) + self.ca(x)

        return out


class CA_layer(nn.Module):
    def __init__(self, channels_in, channels_out, reduction, deg_fsize=512):
        super(CA_layer, self).__init__()
        self.channels_in = channels_in
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.adaptive_channels = nn.Conv2d(deg_fsize, self.channels_in, 1, 1, 0, bias=False)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels_in, channels_in//reduction, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_in // reduction, channels_out, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        # ######################
        # assert len(x[1]) == 1
        # x[1] = x[1][0]
        # #######################
        feature = self.avgpool(x[1])
        
        feature = self.adaptive_channels(feature)
        # self.adaptive_size = nn.Linear(feature.size(1), self.channels_in, bias=False,device='cuda')
        # # breakpoint()
        # feature = self.adaptive_size(feature)  
        
        att = self.conv_du(feature) 
        # breakpoint()

        return x[0] * att


class DAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction):
        super(DAB, self).__init__()

        self.da_conv1 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        self.da_conv2 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.conv2 = conv(n_feat, n_feat, kernel_size)

        self.relu =  nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''

        out = self.relu(self.da_conv1(x))
        out = self.relu(self.conv1(out))
        out = self.relu(self.da_conv2([out, x[1]]))
        out = self.conv2(out) + x[0]

        return out


class DAG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_blocks):
        super(DAG, self).__init__()
        self.n_blocks = n_blocks
        modules_body = [
            DAB(conv, n_feat, kernel_size, reduction) \
            for _ in range(n_blocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        res = x[0]
        for i in range(self.n_blocks):
            res = self.body[i]([res, x[1]])
        res = self.body[-1](res)
        res = res + x[0]

        return res


@BACKBONES.register_module()
class HASR(nn.Module):
    def __init__(   self,
                    in_channels,
                    out_channels,
                    kernel_size = 3,
                    mid_channels = 64,
                    num_blocks = 5,
                    num_groups = 5,
                    upscale_factor = 4,
                    num_layers = 8,
                    scale = 4,
                    reduction = 8,
                    frozen_groups = 0,
                    # deg_rand = False,
                ):
        super(HASR, self).__init__()
        self.num_groups = num_groups
        self.num_blocks = num_blocks
        self.frozen_groups = frozen_groups
        # self.deg_rand = deg_rand
        conv = common.default_conv
        # self.deg_head = build_backbone(deg_head)
        
        modules_head = [conv(in_channels, mid_channels)]
        self.head = nn.Sequential(*modules_head)
        modules_body = [
                DAG(conv, mid_channels, kernel_size, reduction, self.num_blocks) \
                for _ in range(self.num_groups)
            ]
        modules_body.append(conv(mid_channels, mid_channels, kernel_size))
        self.body = nn.Sequential(*modules_body)
        
        modules_tail = [common.Upsampler(conv, scale, mid_channels, act=False),
                        conv(mid_channels, in_channels, kernel_size)] #################
        self.tail = nn.Sequential(*modules_tail)
        self._freeze_stages()
        # breakpoint()
    def forward(self, x, feat):
        # create DASR 
        k_v = feat[0]
        x = self.head(x) 
        res = x 
        # breakpoint()
        for i in range(self.num_groups):
            res = self.body[i]([res, k_v])
            # breakpoint()
            # print(i)
        # print(self.body[-1], "111111111111111111111")
        # breakpoint()
        res = self.body[-1](res)
        res = res + x
        x = self.tail(res)
        return x
    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            pass  # use default initialization
        else:
            raise TypeError('"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')        
        
    def _freeze_stages(self):
        if self.frozen_groups > 0:
            for l in self.head:
                l.eval()
                for param in l.parameters():
                    param.requires_grad = False
        for i in range(0, self.frozen_groups):
            m = self.body[i]
            m.eval()
            for param in m.parameters(): 
                param.requires_grad = False
                # breakpoint()
   
 