import torch
import torch.nn as nn
# from mmedit.models.registry import BACKBONES
from ..builder import BACKBONES
from mmcv.runner import BaseModule
@BACKBONES.register_module()
class EasyRes(BaseModule):
    
    def __init__(self, 
                 in_channels,
                 # out_channels,   
                 frozen_stages = -1,   
                 init_cfg=None,
                 ):
        
        super(EasyRes, self).__init__(init_cfg)
        self.frozen_stages = frozen_stages
        self.E = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),#####################
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        # self.mlp = nn.Sequential(
        #     nn.Linear(256, 256),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(256, 256),
        # )
        self._freeze_stages()
    def forward(self, x):
        
        # breakpoint()
        #############################revised for simclr########################
        if isinstance(x, list):
            x = torch.cat(x)
        #######################################################################        
        fea = self.E(x)
        # out = self.mlp(fea)
    
        return [fea]
    
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.E.parameters():
                param.requires_grad = False
                


if __name__ == "__main__":
    # from mmedit.models.backbones.sr_backbones.ha_edsr import HAEDSR
    import torch
    net = EasyRes(3)
    img = torch.rand([4,3,48,48])
    f= net(img)
    print(f.shape)
    
    
    
    
    