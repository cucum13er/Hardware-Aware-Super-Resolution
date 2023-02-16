# fcdd 
import torch
import torch.nn as nn
from ..builder import BACKBONES
from mmcv.runner import BaseModule
from mmselfsup.models.utils import FCDDNet

@BACKBONES.register_module()
class FCDD_Rui(FCDDNet, BaseModule):
    def __init__(self, in_shape, **kwargs):
        super().__init__(in_shape, **kwargs)
        self.E = nn.Sequential(
            self._create_conv2d(in_shape[0], 64, 5, bias=self.bias, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            self._create_conv2d(64, 64, 5, bias=self.bias, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            self._create_conv2d(64, 128, 5, bias=self.bias, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True), 
            self._create_conv2d(128, 128, 5, bias=self.bias, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            self._create_conv2d(128, 256, 5, bias=self.bias, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            self._create_conv2d(256, 512, 5, bias=self.bias, padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
            )
        
    
    def forward(self, x, ad=True):
        if isinstance(x, list):
            x = torch.cat(x)
        x = self.E(x)

        return [x]
     


if __name__ == "__main__":
    # from mmedit.models.backbones.sr_backbones.ha_edsr import HAEDSR
    import torch
    net = FCDD_Rui((3,48,48))
    img = torch.rand([4,3,48,48])
    f= net(img)
    print(f[0].shape)
    
    
    
    
    