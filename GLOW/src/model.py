import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

#from torch.utils.checkpoint import checkpoint

class ActNorm(nn.Module):
    def __init__(self, channels):
        super(ActNorm, self).__init__()
        self.uninitialized = True
        self.s = nn.Parameter(torch.ones(1,channels,1,1))
        self.b = nn.Parameter(torch.zeros(1,channels,1,1))

    #def initialize(self, x):
    #    '''
    #    Write more here
    #    '''
    #    per_channel = x.transpose(0,1).flatten(1)
    #    channels = per_channel.shape[0]
    #    means = -per_channel.mean(1)
    #    stds = 1/(per_channel.std(1) + 1e-6)
    #    means = means.view(1, channels, 1, 1)
    #    stds = stds.view(1, channels, 1, 1)
    #    
    #    self.register_parameter('b', nn.Parameter(means))
    #    self.register_parameter('s', nn.Parameter(stds))
    #    self.uninitialized = False

    #@torch.cuda.amp.autocast()
    def forward(self, x):
        if self.uninitialized:
            #self.initialize(x)
            per_channel = x.transpose(0,1).flatten(1)
            channels = per_channel.shape[0]
            means = -per_channel.mean(1)
            stds = 1/(per_channel.std(1) + 1e-6)

            means = means.view(1, channels, 1, 1)
            stds = stds.view(1, channels, 1, 1)

            self.b.data.copy_(means)
            self.s.data.copy_(stds)
            self.uninitialized = False 


        y = x*self.s + self.b
        log_det = x.shape[2] * x.shape[3] * torch.log(self.s.abs()).sum()
        return y, log_det

    #@torch.cuda.amp.autocast()
    def backward(self, y):
        ''' inverse of forward '''
        #if self.uninitialized:
        #    self.initialize(y)
        x = (y - self.b) / self.s
        log_det = -y.shape[2] * y.shape[3] * torch.log(self.s.abs()).sum()
        return x, log_det

class CouplingFunction(nn.Module):
    def __init__(self, channels):
        super(CouplingFunction, self).__init__()
        self.channels = channels
        self.conv_layer1 = nn.Conv2d(channels//2, 512, kernel_size=3, padding=1)
        self.conv_layer2 = nn.Conv2d(512, 512, kernel_size=1, padding=1)
        self.conv_layer3 = nn.Conv2d(512, channels, kernel_size=3)

        self.scale_factor = nn.Parameter(torch.zeros(channels,1,1))

        self.actn1 = ActNorm(512)
        self.actn2 = ActNorm(512)

        # We initialize with zeros according to the documentation for better
        # training
        self.conv_layer3.weight.data.zero_()
        self.conv_layer3.bias.data.zero_()

    #@torch.cuda.amp.autocast()
    def forward(self, x):
        z = self.conv_layer3(torch.relu(
                             self.actn2( self.conv_layer2( torch.relu(
                                 self.actn1(self.conv_layer1(x))[0])))[0]))
        z = z*torch.exp(self.scale_factor)

        # Want half channels for s and the other for t
        s = z[:, :self.channels//2, :, :]
        t = z[:, self.channels//2:, :, :]
        return torch.sigmoid(s+2.), t

class AffineCouplingLayer(nn.Module):
    '''
    y1 = s*x1 + t
    y2 = x2
    Coupling functions are 3 layer CNN networks
    '''
    def __init__(self, channels):
        super(AffineCouplingLayer, self).__init__()
        self.coupling = CouplingFunction(channels)

    #@torch.cuda.amp.autocast()
    def forward(self, x):
        x1, x2 = x.chunk(2,1)
        s, t = self.coupling(x2)

        y1 = s*x1 + t
        y2 = x2
        log_det = torch.log(s).sum([1,2,3])
        y = torch.cat((y1, y2), dim=1)
        return y, log_det

    #@torch.cuda.amp.autocast()
    def backward(self, y):
        y1, y2 = y.chunk(2,1)
        s, t = self.coupling(y2)
        x1 = (y1 - t)/s
        x2 = y2
        log_det = - torch.log(s).sum([1,2,3])
        x = torch.cat((x1,x2), dim=1)
        return x, log_det

class AdditiveCouplingLayer(nn.Module):
    '''
    Affine coupling from NICE paper
    y1 = x1 + t
    y2 = x2
    '''
    def __init__(self, channels):
        super(AdditiveCouplingLayer, self).__init()
        self.coupling = CouplingFunction(channels)

    #@torch.cuda.amp.autocast()
    def forward(self, x):
        x1, x2 = x.chunk(2,1)
        s, t = self.coupling(x2)

        y1 = x1 + t
        y2 = x2
        log_det = torch.zeros(x.shape[0])
        y = torch.cat((y1,y2),dim=1)
        return y, log_det

    def backward(self, y):
        y1, y2 = y.chunk(2,1)
        #s, t = y.chunk(2,1)
        s,t = self.coupling(y2)
        x1 = y1 - t
        x2 = y2
        log_det = torch.zeros(y.shape[0])
        y = torch.cat((x1,x2), dim=1)
        return y, log_det

class InvertibleConvolution(nn.Module):
    def __init__(self, channels):
        super(InvertibleConvolution, self).__init__()
        self.w = nn.Parameter(torch.nn.init.orthogonal_(
                                            torch.randn(channels, channels)))

    #@torch.cuda.amp.autocast()
    def forward(self, x):
        #b, a, c, d = x.shape
        y = F.conv2d(x, self.w.unsqueeze(-1).unsqueeze(-1))
        log_det = x.shape[2] * x.shape[3] * torch.slogdet(self.w)[1]
        return y, log_det

    #@torch.cuda.amp.autocast()
    def backward(self, y):
        w_inv = self.w.inverse()
        # Direct way
        #s = torch.sum(torch.log(torch.abs(torch.prod(torch.diagonal(U)))))
        x = F.conv2d(y, w_inv.unsqueeze(-1).unsqueeze(-1))
        s = torch.slogdet(self.w)[1]
        #log_det = -y.shape[2] * y.shape[3] * torch.slogdet(self.w)[1]

        # LU Decomp
        #LU, pivots = torch.lu(self.w)
        #P, L, U = torch.lu_unpack(LU, pivots)
        #s = torch.sum(torch.log(torch.abs(torch.diagonal(U))))
        log_det = -y.shape[2] * y.shape[3] * s
        return x, log_det

class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    #@torch.cuda.amp.autocast()
    def forward(self, x):
        _, C, H, W = x.shape
        x = x.reshape(-1, C, H//2, 2, W//2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(-1, 4*C, H//2, W//2)
        return x

    #@torch.cuda.amp.autocast()
    def backward(self, y):
        _, C, H, W = y.shape
        y = y.reshape(-1, C//4, 2, 2, H, W)
        y = y.permute(0, 1, 4, 2, 5, 3)
        y = y.reshape(-1, C//4, 2*H, 2*W)
        return y

class Split(nn.Module):
    def __init__(self, channels):
        super(Split, self).__init__()
        self.mean = nn.Conv2d(channels//2, 
                              channels//2, 
                              kernel_size=3, 
                              padding=1)
        self.log_std = nn.Conv2d(channels//2,
                                 channels//2, 
                                 kernel_size=3,
                                 padding=1)

        self.mean.weight.data.zero_()
        self.mean.bias.data.zero_()
        self.log_std.weight.data.zero_()
        self.log_std.bias.data.zero_()

    #@torch.cuda.amp.autocast()
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        mu = self.mean(x1)
        l_std = self.log_std(x1)
        z2 = (x2 - mu) * torch.exp(-l_std)
        log_det = -l_std.sum([1,2,3])
        return x1, z2, log_det

    #@torch.cuda.amp.autocast()
    def backward(self, x1, z2):
        mu = self.mean(x1)
        l_std = self.log_std(x1)
        x2 = z2 * torch.exp(l_std) - mu
        log_det = -l_std.sum([1,2,3])
        y = torch.cat([x1, x2], dim=1)
        return y, log_det

class GLOWStep(nn.Module):
    '''
    ActNorm
    Invertible 1x1 Convolution
    Coupling
    '''
    def __init__(self, channels, additive=False):
        super(GLOWStep, self).__init__()
        self.actnorm = ActNorm(channels)
        self.invert_conv = InvertibleConvolution(channels)
        if additive:
            self.affine = AdditiveCouplingLayer(channels)
        else:
            self.affine = AffineCouplingLayer(channels)

    #@torch.cuda.amp.autocast()
    def forward(self, x):
        log_det_sum = 0.
        # ActNorm
        y, log_det = self.actnorm(x)
        log_det_sum = log_det_sum + log_det
        # Invert 1x1
        y, log_det = self.invert_conv(y)
        log_det_sum = log_det_sum + log_det
        # Affine
        y, log_det = self.affine(y)
        log_det_sum = log_det_sum + log_det

        return y, log_det_sum

    #@torch.cuda.amp.autocast()
    def backward(self, y):
        log_det_sum = 0
        # Affine
        x, log_det = self.affine.backward(y)
        log_det_sum = log_det_sum + log_det
        # Invert 1x1
        x, log_det = self.invert_conv.backward(x)
        log_det_sum += log_det
        # ActNorm
        x, log_det = self.actnorm.backward(x)
        log_det_sum += log_det

        return x, log_det_sum

class GLOWLevel(nn.Module):
    '''
    Squeeze
    k GLOW Steps
    Split
    '''
    def __init__(self, channels, depth=32, additive=False):
        super(GLOWLevel, self).__init__()
        self.squeeze = Squeeze()
        self.steps = nn.ModuleList(
                [GLOWStep(4*channels, additive) for _ in range(depth)])
        self.split = Split(4*channels)

    #@torch.cuda.amp.autocast()
    def forward(self, x):
        x = self.squeeze(x)
        log_det_sum = 0.
        # K Steps
        for step in self.steps:
            x, log_det = step(x)
            log_det_sum = log_det_sum + log_det
        # Split
        x1, z2, log_det  = self.split(x)
        log_det_sum += log_det
        return x1, z2, log_det_sum

    #@torch.cuda.amp.autocast()
    def backward(self, x1, z2):
        y, log_det_sum = self.split.backward(x1,z2)
        for step in reversed(self.steps):
            y, log_det = step.backward(y)
            log_det_sum += log_det
        y = self.squeeze.backward(y)
        return y, log_det_sum

class GLOW(nn.Module):
    def __init__(self, channels, depth, n_levels, additive=False):
        super(GLOW, self).__init__()
        self.levels = nn.ModuleList(
                #[GLOWLevel(channels*2**i, depth, additive) for i in range(n_levels-1)])
                [GLOWLevel(channels*2**i, depth, additive) for i in range(n_levels-1)])
        self.squeeze = Squeeze()
        last_n_channels = 4*(channels*2**(n_levels-1))
        self.steps = nn.ModuleList(
                [GLOWStep(last_n_channels, additive) for _ in range(depth)])

        self.mean = nn.Conv2d(last_n_channels, last_n_channels, kernel_size=3, padding=1)
        self.log_std = nn.Conv2d(last_n_channels, last_n_channels, kernel_size=3, padding=1)

        self.mean.weight.data.zero_()
        self.mean.bias.data.zero_()
        self.log_std.weight.data.zero_()
        self.log_std.bias.data.zero_()

    #@torch.cuda.amp.autocast()
    def forward(self, x):
        z_list = []
        log_det_sum = 0.
        # Through levels
        #x.requires_grad=True
        for i,level in enumerate(self.levels):
            x, z, log_det = level(x)
            #x, z, log_det = checkpoint(level, x)
            log_det_sum += log_det
            z_list.append(z)
        # Squeeze
        x = self.squeeze(x)
        # GLOW Steps without split
        # Extra steps after all levels
        for i,step in enumerate(self.steps):
            x, log_det = step(x)
            log_det_sum += log_det
        # As Gaussians on last z variables
        mock_vals = torch.zeros_like(x)
        mu = self.mean(mock_vals)
        l_std = self.log_std(mock_vals)

        z = (x - mu) * torch.exp(-l_std)
        log_det = -l_std.sum()
        log_det_sum += log_det

        z_list.append(z)
        return z_list, log_det_sum

    #@torch.cuda.amp.autocast()
    def backward(self, z_list):
        print("Backwards pass")
        z = z_list[-1] # Get prior

        # Gaussians
        mock_vals = torch.zeros_like(z)
        mu = self.mean(mock_vals)
        l_std = self.log_std(mock_vals)

        log_det_sum = 0.

        x = z * torch.exp(l_std) + mu
        #x.requires_grad=True
        log_det = l_std.sum()
        log_det_sum = log_det_sum + log_det

        # Last steps
        for i, step in enumerate(reversed(self.steps)):
            x, log_det = step.backward(x)
            #x, log_det = checkpoint(step.backward, x)
            log_det_sum = log_det_sum + log_det
        # Last Squeeze
        x = self.squeeze.backward(x)
        # Levels in Reverse
        for i, level in enumerate(reversed(self.levels)):
            z = z_list[-(2+i)]
            x, log_det = level.backward(x, z)
            log_det_sum = log_det_sum + log_det

        return x, log_det_sum
