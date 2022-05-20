
import mmedit.models.backbones.sr_backbones.common as common
import torch.nn.functional as F
#from option import args as args_global
import torch
from mmcv.runner import load_checkpoint
from torch import nn
from mmedit.models.builder import build_backbone
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


# class DA_conv(nn.Module):
#     def __init__(self, channels_in, channels_out, kernel_size, reduction, deg_fsize=512):
#         super(DA_conv, self).__init__()
#         self.channels_out = channels_out
#         self.channels_in = channels_in
#         self.kernel_size = kernel_size
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.adaptive_size = nn.Linear(deg_fsize, self.channels_in, bias=False)
#         #revise the kernel into adaptive to input parameters 
#         self.kernel = nn.Sequential(
#             nn.Linear(self.channels_in, self.channels_in//reduction, bias=False),
#             nn.LeakyReLU(0.1, True),
#             nn.Linear(self.channels_in//reduction, self.channels_out * self.kernel_size * self.kernel_size, bias=False)
#         )
#         self.conv = nn.Conv2d(channels_in, channels_out, 1, padding=1//2, bias=True)
#         # self.conv = common.default_conv(channels_in, channels_out, 1)
#         self.ca = CA_layer(channels_in, channels_out, reduction)
        
#         self.relu = nn.LeakyReLU(0.1, True)

#     def forward(self, x):
#         '''
#         :param x[0]: feature map: B * C * H * W
#         :param x[1]: degradation representation: B * C
#         '''
#         # print(x[0].size())
#         b, c, h, w = x[0].size()
        
#         # breakpoint()
#         # branch 1
#         feature = self.avgpool(x[1]) # b*2048*1*1
#         feature = feature.view(feature.size(0), -1) # b*2048
#         # self.adaptive_size = nn.Linear(feature.size(1), self.channels_in, bias=False,device='cuda')
#         # breakpoint()
#         feature = self.adaptive_size(feature)
#         kernel = self.kernel(feature).view(-1, 1, self.kernel_size, self.kernel_size)
#         out = self.relu(F.conv2d(x[0].view(1, -1, h, w), kernel, groups=b*c, padding=(self.kernel_size-1)//2))
#         out = self.conv(out.view(b, -1, h, w))
#         # breakpoint()
#         # branch 2
#         out = out + self.ca(x)

#         return out


# class CA_layer(nn.Module):
#     def __init__(self, channels_in, channels_out, reduction, deg_fsize=512):
#         super(CA_layer, self).__init__()
#         self.channels_in = channels_in
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.adaptive_channels = nn.Conv2d(deg_fsize, self.channels_in, 1, 1, 0, bias=False)
#         self.conv_du = nn.Sequential(
#                 nn.Conv2d(channels_in, channels_in // reduction, 1, padding=0, bias=True),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(channels_in // reduction, channels_out, 1, padding=0, bias=True),
#                 nn.Sigmoid()
#         )

#     def forward(self, x):
#         '''
#         :param x[0]: feature map: B * C * H * W
#         :param x[1]: degradation representation: B * C
#         '''
#         # ######################
#         # assert len(x[1]) == 1
#         # x[1] = x[1][0]
#         # #######################
#         feature = self.avgpool(x[1])
        
#         feature = self.adaptive_channels(feature)
#         # self.adaptive_size = nn.Linear(feature.size(1), self.channels_in, bias=False,device='cuda')
#         # # breakpoint()
#         # feature = self.adaptive_size(feature)  
        
#         att = self.conv_du(feature) 
#         # breakpoint()

#         return x[0] * att

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res    
# class DAB(nn.Module):
#     def __init__(self, conv, n_feat, kernel_size, reduction):
#         super(DAB, self).__init__()

#         self.da_conv1 = DA_conv(n_feat, n_feat, kernel_size, reduction)
#         self.da_conv2 = DA_conv(n_feat, n_feat, kernel_size, reduction)
#         self.conv1 = conv(n_feat, n_feat, kernel_size)
#         self.conv2 = conv(n_feat, n_feat, kernel_size)

#         self.relu =  nn.LeakyReLU(0.1, True)

#     def forward(self, x):
#         '''
#         :param x[0]: feature map: B * C * H * W
#         :param x[1]: degradation representation: B * C
#         '''

#         out = self.relu(self.da_conv1(x))
#         out = self.relu(self.conv1(out))
#         out = self.relu(self.da_conv2([out, x[1]]))
#         out = self.conv2(out) + x[0]

#         return out


# class DAG(nn.Module):
#     def __init__(self, conv, n_feat, kernel_size, reduction, n_blocks):
#         super(DAG, self).__init__()
#         self.n_blocks = n_blocks
#         modules_body = [
#             DAB(conv, n_feat, kernel_size, reduction) \
#             for _ in range(n_blocks)
#         ]
#         modules_body.append(conv(n_feat, n_feat, kernel_size))

#         self.body = nn.Sequential(*modules_body)

#     def forward(self, x):
#         '''
#         :param x[0]: feature map: B * C * H * W
#         :param x[1]: degradation representation: B * C
#         '''
#         res = x[0]
#         for i in range(self.n_blocks):
#             res = self.body[i]([res, x[1]])
#         res = self.body[-1](res)
#         res = res + x[0]

#         return res
## Residual Channel Attention Network (RCAN)
class HARCAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(HARCAN, self).__init__()
        
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale[0]
        act = nn.ReLU(True)
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

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

# @BACKBONES.register_module()
# class HARCAN(nn.Module):
#     def __init__(   self,
#                     deg_head,
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
#                     deg_rand = False,
#                 ):
#         super(HARCAN, self).__init__()
#         self.num_groups = num_groups
#         self.num_blocks = num_blocks
#         self.deg_rand = deg_rand
#         conv = common.default_conv
#         self.deg_head = build_backbone(deg_head)
        
#         modules_head = [conv(in_channels, mid_channels)]
#         self.head = nn.Sequential(*modules_head)
#         modules_body = [
#                 DAG(conv, mid_channels, kernel_size, reduction, self.num_blocks) \
#                 for _ in range(self.num_groups)
#             ]
#         modules_body.append(conv(mid_channels, mid_channels, kernel_size))
#         self.body = nn.Sequential(*modules_body)
        
#         modules_tail = [common.Upsampler(conv, scale, mid_channels, act=False),
#                         conv(mid_channels, in_channels, kernel_size)] #################
#         self.tail = nn.Sequential(*modules_tail)
#         # breakpoint()
#     def forward(self, x):
#         # get the degradation feature map
#         if self.deg_rand:
#             b, _, _, _ = x.size()
#             k_v = torch.rand(b,512,1,1, device='cuda')
#             # from scipy.io import savemat
#             # results = {"img": k_v.to('cpu').numpy()}
#             # savemat("work_dirs/restorers/dasr/dasr_x4c64b16_g1_100k_div2k_debug/matlab_matrix"+".mat", results)            
#         else:
#             k_v = self.deg_head(x)[0]
#             # from scipy.io import savemat
#             # results = {"img": k_v.to('cpu').numpy()}
#             # savemat("work_dirs/restorers/dasr/dasr_x4c64b16_g1_100k_div2k_debug/matlab_matrix"+".mat", results)
        
#         #breakpoint()
#         # create DASR 
#         x = self.head(x) 
#         res = x 
#         for i in range(self.num_groups):
#             res = self.body[i]([res, k_v])
#             # print(i)
#         # print(self.body[-1], "111111111111111111111")
#         # breakpoint()
#         res = self.body[-1](res)
#         res = res + x
#         x = self.tail(res)
#         return x
#     def init_weights(self, pretrained=None, strict=True):
#         """Init weights for models.

#         Args:
#             pretrained (str, optional): Path for pretrained weights. If given
#                 None, pretrained weights will not be loaded. Defaults to None.
#             strict (boo, optional): Whether strictly load the pretrained model.
#                 Defaults to True.
#         """
#         if isinstance(pretrained, str):
#             logger = get_root_logger()
#             load_checkpoint(self, pretrained, strict=strict, logger=logger)
#         elif pretrained is None:
#             pass  # use default initialization
#         else:
#             raise TypeError('"pretrained" must be a str or None. '
#                             f'But received {type(pretrained)}.')        
        
    # def __init__(self, args, conv=common.default_conv):
    #     super(DASR, self).__init__()

    #     self.n_groups = 5
    #     n_blocks = 5
    #     n_feats = 64
    #     kernel_size = 3
    #     reduction = 8
    #     scale = int(args.scale[0])

    #     # RGB mean for DIV2K
    #     rgb_mean = (0.4488, 0.4371, 0.4040)
    #     rgb_std = (1.0, 1.0, 1.0)
    #     self.sub_mean = common.MeanShift(255.0, rgb_mean, rgb_std)
    #     self.add_mean = common.MeanShift(255.0, rgb_mean, rgb_std, 1)

    #     # head module
    #     modules_head = [conv(in_channels, channel_growth)] ###################
    #     self.head = nn.Sequential(*modules_head)

    #     # compress
    #     self.compress = nn.Sequential(
    #         nn.Linear(256, 64, bias=False),
    #         nn.LeakyReLU(0.1, True)
    #     )

    #     # body
    #     modules_body = [
    #         DAG(common.default_conv, n_feats, kernel_size, reduction, n_blocks) \
    #         for _ in range(self.n_groups)
    #     ]
    #     modules_body.append(conv(n_feats, n_feats, kernel_size))
    #     self.body = nn.Sequential(*modules_body)

    #     # tail
    #     modules_tail = [common.Upsampler(conv, scale, n_feats, act=False),
    #                     conv(n_feats, args_global.n_colors, kernel_size)] #################
    #     self.tail = nn.Sequential(*modules_tail)

    # def forward(self, x, k_v):
    #     k_v = self.compress(k_v)

    #     # sub mean
    #     x = self.sub_mean(x)

    #     # head
    #     x = self.head(x)

    #     # body
    #     res = x
    #     for i in range(self.n_groups):
    #         res = self.body[i]([res, k_v])
    #     res = self.body[-1](res)
    #     res = res + x

    #     # tail
    #     x = self.tail(res)

    #     # add mean
    #     x = self.add_mean(x)

        # return x


# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()

#         self.E = nn.Sequential(
#             nn.Conv2d(args_global.n_colors, 64, kernel_size=3, padding=1),#####################
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.1, True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.1, True),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.1, True),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.1, True),
#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.1, True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.1, True),
#             nn.AdaptiveAvgPool2d(1),
#         )
#         self.mlp = nn.Sequential(
#             nn.Linear(256, 256),
#             nn.LeakyReLU(0.1, True),
#             nn.Linear(256, 256),
#         )

#     def forward(self, x):
#         fea = self.E(x).squeeze(-1).squeeze(-1)
#         out = self.mlp(fea)

#         return fea, out


# class BlindSR(nn.Module):
#     def __init__(self, args):
#         super(BlindSR, self).__init__()

#         # Generator
#         self.G = DASR(args)

#         # Encoder
#         self.E = MoCo(base_encoder=Encoder)

#     def forward(self, x):
#         if self.training:
#             x_query = x[:, 0, ...]                          # b, c, h, w
#             x_key = x[:, 1, ...]                            # b, c, h, w

#             # degradation-aware represenetion learning
#             fea, logits, labels = self.E(x_query, x_key)

#             # degradation-aware SR
#             sr = self.G(x_query, fea)

#             return sr, logits, labels
#         else:
#             # degradation-aware represenetion learning
#             fea = self.E(x, x)

#             # degradation-aware SR
#             sr = self.G(x, fea)

#             return sr
