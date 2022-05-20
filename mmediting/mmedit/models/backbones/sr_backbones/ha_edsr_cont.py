# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint
from mmedit.models.builder import build_backbone
from mmedit.models.common import (PixelShufflePack, default_init_weights,
                                  make_layer)
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)

    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    
    assert (content_feat.size()[0] == style_feat.size()[0])
    assert (content_feat.size()[1] * 2 == style_feat.size()[1])
    size = content_feat.size()
    style_channels = style_feat.size(1)
    style_mean = style_feat[:,:style_channels//2,:,:]
    style_std = style_feat[:,style_channels//2:,:,:]
    # style_mean, style_std = calc_mean_std(style_feat)
    # print(style_mean,style_std)
    content_mean, content_std = calc_mean_std(content_feat)
    # print(content_mean,content_std)
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    # print(normalized_feat)
    # breakpoint()
    # without normalization
    
    
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

class ResidualBlock_AdaIN(nn.Module):
    def __init__(self, mid_channels=64, res_scale=1.0, featsize=512):
        super().__init__()
        assert featsize == mid_channels * 8
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.AdaNN = nn.Sequential(
                    nn.Linear(featsize, featsize//2, bias=False),
                    nn.LeakyReLU(0.1, True),
                    nn.Linear(featsize//2, featsize//4, bias=False),
                    )
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

        # if res_scale < 1.0, use the default initialization, as in EDSR.
        # if res_scale = 1.0, use scaled kaiming_init, as in MSRResNet.
        if res_scale == 1.0:
            self.init_weights()
    def forward(self, x, features):
        # x is a list
        # x[0] is content, x[1] is features
        # b*64*H*W -> b*64*1*1 -> b*64
        # breakpoint()
        # print(len(x))
        # assert len(x) == 2
        # features = x[1]
        # x = x[0]
        features = self.avgpool(features)
        features = features.squeeze(-1).squeeze(-1)
        features = self.AdaNN( features ).unsqueeze(-1).unsqueeze(-1)
        
        # b*64*H*W
        identity = x
        # b*64*H*W
        out = self.conv2(self.relu(self.conv1(x)))      
        # breakpoint()
        out = adaptive_instance_normalization(out, features)
        
        return identity + self.res_scale * out
    
    def init_weights(self):
        """Initialize weights for ResidualBlockNoBN.

        Initialization methods like `kaiming_init` are for VGG-style
        modules. For modules with residual paths, using smaller std is
        better for stability and performance. We empirically use 0.1.
        See more details in "ESRGAN: Enhanced Super-Resolution Generative
        Adversarial Networks"
        """

        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)


class ResidualBlock_Att(nn.Module):
    def __init__(self, mid_channels=64, res_scale=1.0, featsize=512):
        super().__init__()
        assert featsize == mid_channels * 8
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.AdaNN = nn.Sequential(
                    nn.Linear(featsize, featsize//4, bias=False),
                    nn.LeakyReLU(0.1, True),
                    nn.Linear(featsize//4, featsize//8, bias=False),
                    )
        self.res_scale = res_scale
        # self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        # self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        # self.relu = nn.ReLU(inplace=True)

        # if res_scale < 1.0, use the default initialization, as in EDSR.
        # if res_scale = 1.0, use scaled kaiming_init, as in MSRResNet.
        # if res_scale == 1.0:
        #     self.init_weights()
    def forward(self, x, features):
        
        features = self.avgpool(features).squeeze(-1).squeeze(-1)
        features = self.AdaNN(features).unsqueeze(-1).unsqueeze(-1)
        identity = x
        # breakpoint()
        out = torch.multiply(x, features)
        
        
        return identity + self.res_scale * out
    
    # def init_weights(self):
    #     """Initialize weights for ResidualBlockNoBN.

    #     Initialization methods like `kaiming_init` are for VGG-style
    #     modules. For modules with residual paths, using smaller std is
    #     better for stability and performance. We empirically use 0.1.
    #     See more details in "ESRGAN: Enhanced Super-Resolution Generative
    #     Adversarial Networks"
    #     """

    #     for m in [self.conv1, self.conv2]:
    #         default_init_weights(m, 0.1)
            
            
            
            
class UpsampleModule(nn.Sequential):
    """Upsample module used in EDSR.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        mid_channels (int): Channel number of intermediate features.
    """

    def __init__(self, scale, mid_channels):
        modules = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                modules.append(
                    PixelShufflePack(
                        mid_channels, mid_channels, 2, upsample_kernel=3))
        elif scale == 3:
            modules.append(
                PixelShufflePack(
                    mid_channels, mid_channels, scale, upsample_kernel=3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')

        super().__init__(*modules)


@BACKBONES.register_module()
class HAEDSR_Cont(nn.Module):
    """EDSR network structure.

    Paper: Enhanced Deep Residual Networks for Single Image Super-Resolution.
    Ref repo: https://github.com/thstkdgus35/EDSR-PyTorch

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        num_blocks (int): Block number in the trunk network. Default: 16.
        upscale_factor (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
        rgb_std (tuple[float]): Image std in RGB orders. In EDSR, it uses
            (1.0, 1.0, 1.0). Default: (1.0, 1.0, 1.0).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 # deg_head=None,
                 mid_channels=64,
                 num_blocks=16,
                 upscale_factor=4,
                 res_scale=1,
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 rgb_std=(1.0, 1.0, 1.0),
                 # deg_rand = False,
                 ):
        super().__init__()
        # self.deg_head = build_backbone(deg_head) if deg_head else None
        # self.deg_rand = deg_rand
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.num_blocks = num_blocks
        self.upscale_factor = upscale_factor

        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.std = torch.Tensor(rgb_std).view(1, 3, 1, 1)

        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        # self.body = make_layer(
        #     ResidualBlock_AdaIN,
        #     num_blocks=1,
        #     mid_channels=mid_channels,
        #     res_scale=res_scale,
        #     featsize=512,
        #     ) # unable to be sequential 
        self.body = ResidualBlock_AdaIN(mid_channels,res_scale,featsize=512)
        # self.body = ResidualBlock_Att(mid_channels,res_scale,featsize=512)
        self.conv_after_body = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.upsample = UpsampleModule(upscale_factor, mid_channels)
        self.conv_last = nn.Conv2d(
            mid_channels, out_channels, 3, 1, 1, bias=True)

    def forward(self, x, feat):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # b, _, _, _ = x.size()
        # if self.deg_rand == True:
        #     # b, _, _, _ = x.size()
        #     features = torch.rand(b,512,1,1, device='cuda')
        # else:
        #     features = self.deg_head(x)
        #     features = features[0] #+ torch.rand(b,512,1,1, device='cuda')
        features = feat[0]
        
        self.mean = self.mean.to(x)
        self.std = self.std.to(x)

        x = (x - self.mean) / self.std
        x = self.conv_first(x)
        # breakpoint()
        #x = self.body([x, features])
        for i in range(self.num_blocks):
            # print(i,'11111111111111111111111')

            x = self.body(x,features)
            # breakpoint()
        res = self.conv_after_body(x)
        res += x

        x = self.conv_last(self.upsample(res))
        x = x * self.std + self.mean
        # print(self.conv_last.weight.grad,'1111111')
        # breakpoint()
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


if __name__ == "__main__":
    # from mmedit.models.backbones.sr_backbones.ha_edsr import HAEDSR
    feat = torch.rand(16,512,2,2)
    x = torch.rand(16,64,49,49)
    Net = ResidualBlock_AdaIN()
    Net(x, feat)
    
    
    
    
    
    
    
    
    