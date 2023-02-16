import torch
from mmcv.runner import load_checkpoint
import torch.nn as nn
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
@BACKBONES.register_module()
class EasyRes(nn.Module):
    
    def __init__(self, 
                 in_channels,
                 # out_channels,
                 frozen_stages = -1,                   
                 pretrained = None,                           
                 ):
        
        super(EasyRes, self).__init__()
        
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
        self.init_weights(pretrained)
        # self.mlp = nn.Sequential(
        #     nn.Linear(256, 256),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(256, 256),
        # )
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
    def forward(self, x):
        #############################revised for simclr########################
        if isinstance(x, list):
            x = torch.cat(x)
        #######################################################################             

        fea = self.E(x)
        # out = self.mlp(fea)
        # breakpoint()
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
    
    
    
    
    