import torch
import torch.nn as nn
import math
import numpy
import utils
import random

def transform(data, msfa_size=4, spatial_ratio=2):
    # rotation
    if numpy.random.uniform() < 0.5:
        rn = random.randint(1, 3)
        data = torch.rot90(data, rn, [2, 3])
 
    # flip
    if numpy.random.uniform() < 0.5:
        r = numpy.random.uniform()
        if r < 0.25:
            data = torch.flip(data, [2])
        elif r < 0.5:
            data = torch.flip(data, [3])
        elif r < 0.75:
            data = torch.flip(data, [2])
            data = torch.flip(data, [3])
        else:
            data = torch.flip(data, [3])
            data = torch.flip(data, [2])

    # shift
    if numpy.random.uniform() < 0.5:
        i = random.randint(0, msfa_size*spatial_ratio-1)
        j = random.randint(0, msfa_size*spatial_ratio-1)
        data = torch.roll(data, (i, j), (2, 3))

    return data

class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        self.D = 3
        self.channels = 64
        self.msfa_size = [4, 4]

        self.conv1 = nn.Conv2d(args.num_bands+1, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, args.num_bands, 3, 1, 1)

        self.relu = nn.ReLU()
    
    def forward(self, mosaic, pan):
        mosaic_reshape = torch.nn.functional.pixel_unshuffle(mosaic, downscale_factor=4)
        mosaic_interpolate = torch.nn.functional.interpolate(mosaic_reshape, scale_factor=8, mode="bilinear")
        x = torch.cat((mosaic_interpolate, pan), 1)
        y = self.conv3(self.relu(self.conv2(self.relu(self.conv1(x))))) + pan.repeat(1, mosaic_interpolate.shape[1], 1, 1)
        return y

class Degrade_R(nn.Module):
    def __init__(self, args):
        super(Degrade_R, self).__init__()

        self.spec_res = nn.Conv2d(args.num_bands, 1, 1, 1, 0, bias=False)
        self.spec_res.weight.data = torch.ones_like(self.spec_res.weight)
        self.spec_res.weight.data /= self.spec_res.weight.data.sum()
    
    def forward(self, hrms):
        y = self.spec_res(hrms)
        return y

def degrade_dm(hrms, msfa_kernel):
    x = torch.nn.functional.conv2d(hrms, msfa_kernel, bias=None, stride=msfa_kernel.shape[2], groups=hrms.shape[1])
    x = torch.nn.functional.pixel_shuffle(x, 4)

    return x



