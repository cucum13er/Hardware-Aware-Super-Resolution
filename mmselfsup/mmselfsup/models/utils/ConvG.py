import torch
import math
from torch import Tensor
from typing import Optional
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

class ConvG(torch.nn.Module):

    def __init__(self,
                 in_channels: int, 
                 out_channels: int,                 
                 kernel_size, 
                 sigma_min= 0.2,
                 sigma_max = 4,
                 stride=1, 
                 padding=0, 
                 dilation=1, 
                 groups=1, 
                 bias=True, 
                 padding_mode='zeros', 
                 device=None, 
                 dtype=None,                 
                ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        weight = torch.empty( (out_channels, in_channels // groups, *kernel_size), **factory_kwargs)
        N, C, h, _ = weight.shape
        sx = np.random.uniform(sigma_min, sigma_max, [N,C])
        sy = np.random.uniform(sigma_min, sigma_max, [N,C])
        theta = np.random.uniform(0, 2*np.pi, [N,C])
        self.weight = Parameter(torch.tensor(np.stack((sx,sy,theta),axis=2), **factory_kwargs) )                   
        if bias:
            self.bias = Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            
    def cal_sigma(self, sig_x, sig_y, radians):
        sig_x = sig_x.view(-1, 1, 1)
        sig_y = sig_y.view(-1, 1, 1)
        radians = radians.view(-1, 1, 1)
    
        D = torch.cat([F.pad(sig_x ** 2, [0, 1, 0, 0]), F.pad(sig_y ** 2, [1, 0, 0, 0])], 1)
        U = torch.cat([torch.cat([radians.cos(), -radians.sin()], 2),
                       torch.cat([radians.sin(), radians.cos()], 2)], 1)
        sigma = torch.bmm(U, torch.bmm(D, U.transpose(1, 2))).cuda()
    
        return sigma  
    
    def anisotropic_gaussian_kernel(self, batch, kernel_size, covar):
        ax = torch.arange(kernel_size).float().cuda() - kernel_size // 2
    
        xx = ax.repeat(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
        yy = ax.repeat_interleave(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
        xy = torch.stack([xx, yy], -1).view(batch, -1, 2)
    
        inverse_sigma = torch.inverse(covar)
        kernel = torch.exp(- 0.5 * (torch.bmm(xy, inverse_sigma) * xy).sum(2)).view(batch, kernel_size, kernel_size)

        return kernel / kernel.sum([1, 2], keepdim=True)   
         
    def stable_anisotropic_gaussian_kernel(self, kernel_size=21, theta=0, lambda_1=0.2, lambda_2=4.0):
        theta = torch.ones(1).cuda()* theta / 180 * math.pi
        lambda_1 = torch.ones(1).cuda() * lambda_1
        lambda_2 = torch.ones(1).cuda() * lambda_2

        covar = self.cal_sigma(lambda_1, lambda_2, theta)
        kernel = self.anisotropic_gaussian_kernel(1, kernel_size, covar)
        return kernel.squeeze()

        
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        # weight: N*C*[sigmax, sigmay, rotation]
        N,C,_ = weight.shape
        weight_final = torch.empty((N, C, self.kernel_size[0], self.kernel_size[1])).cuda()
        for n in range(N):
            for c in range(C):
                sigmax,sigmay,theta = weight[n,c,:]                
                weight_final[n,c,:,:] = self.stable_anisotropic_gaussian_kernel(kernel_size=self.kernel_size[0], 
                                                                           theta=theta, 
                                                                           lambda_1=sigmax, 
                                                                           lambda_2=sigmay)                
        return F.conv2d(input, weight_final, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)
    
    
    
if __name__ == "__main__":
    
    device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = ConvG(3, 16, kernel_size=(3,3), padding=1, device=device)
    w1 = net.weight.clone()
    b1 = net.bias.clone()
    inputs = (torch.randn(4, 3, 8, 8)).to(device)
    gt = torch.randn(4, 16, 8, 8).to(device)
    net.train()

    from torch import optim
    optimizer = optim.SGD(net.parameters(),lr=0.1)
    optimizer.zero_grad()#zeroing out every new batch so it doesnt affect the next batch
    out = net(inputs) #prediction
    criterion = torch.nn.L1Loss(reduction='sum')
    loss=criterion(out,gt)
    loss.backward() #backprop
    optimizer.step() #updates weight and bias using computed gradients
    w2 = net.weight.clone()
    b2 = net.bias
    print('output size: ', out.size())
    print('weights szie: ', w2.size())
    print('weights updates: ', w2-w1)
    print('bias updates:', b2-b1)
        
