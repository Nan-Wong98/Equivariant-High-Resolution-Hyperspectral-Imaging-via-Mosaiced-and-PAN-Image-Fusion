import torch
import torch.nn as nn
import math
import numpy as np
import utils
import time

class Res(nn.Module):
    def __init__(self, channels=32):
        super(Res, self).__init__()
        self.res = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1),
                                 nn.ReLU(),
                                 nn.Conv2d(channels, channels, 3, 1, 1))
    
    def forward(self, x):
        return x + self.res(x)

class SpNet(nn.Module):
    def __init__(self, args):
        super(SpNet, self).__init__()
        self.D = 16
        self.C = 64
        self.msfa_size = args.msfa_size

        self.head = nn.Conv2d(in_channels=args.num_bands, out_channels=self.C, kernel_size=3, stride=1, padding=1)

        self.res_blks = nn.ModuleList([Res(self.C) for _ in range(self.D)])

        self.tail1 = nn.Conv2d(in_channels=self.C, out_channels=self.C, kernel_size=3, stride=1, padding=1)

        self.tail2 = nn.Conv2d(in_channels=self.C, out_channels=args.num_bands, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # t = time.time()
        x_sparse = torch.nn.functional.pixel_unshuffle(x, 4)
        x_sparse = torch.nn.functional.interpolate(x_sparse, scale_factor=4, mode="bicubic")
        x = x_sparse.detach()
        # print(time.time() - t)
        x = self.head(x)
        x_ = x
        for i in range(self.D):
            x = self.res_blks[i](x)

        x = self.tail1(x) + x_
        x = self.tail2(x)
        return x

def hp(x):
    C = x.shape[1]
    kernel = torch.zeros(1, 1, 3, 3).to(x.device)
    kernel[0, 0] = torch.tensor([[1., 1., 1.],
                                [1., -8., 1.],
                                [1., 1., 1.]])
    kernel = kernel.repeat(C, 1, 1, 1)
    y = nn.functional.conv2d(x, kernel, stride=1, padding=3//2, groups=C)

    return y

class Generator(nn.Module): # from https://github.com/yuwei998/PanGAN/blob/master
    def __init__(self, args):
        super(Generator, self).__init__()
        self.scale_factor = args.spatial_ratio
        self.conv1 = nn.Conv2d(args.num_bands+1, 64, 9, 1, 4)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64 + args.num_bands+1, 32, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32+64+args.num_bands+1, args.num_bands, 5, 1, 2)
        self.tanh = nn.Tanh()

    def forward(self, lrms, pan):
        upms = torch.nn.functional.interpolate(lrms, scale_factor=self.scale_factor, mode="bilinear")
        x0 = torch.cat((upms, pan), 1)
        x1 = self.relu1(self.bn1(self.conv1(x0)))
        x2 = self.relu2(self.bn2(self.conv2(torch.cat((x0, x1), 1))))
        x3 = self.tanh(self.conv3(torch.cat((x0, x1, x2), 1)))
        return x3

class Discriminator_spe(nn.Module):
    def __init__(self, args):
        super(Discriminator_spe, self).__init__()
        self.scale_factor = args.spatial_ratio
        self.conv1 = nn.Conv2d(args.num_bands, 16, 3, 2, 1)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(16, 32, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.lrelu2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.lrelu3 = nn.LeakyReLU(0.2)

        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.lrelu4 = nn.LeakyReLU(0.2)

        self.conv5 = nn.Conv2d(128, 1, 4, 4, 1)
        self.lrelu5 = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.bn2(self.conv2(x)))
        x = self.lrelu3(self.bn3(self.conv3(x)))
        x = self.lrelu4(self.bn4(self.conv4(x)))
        x = self.lrelu5(self.conv5(x))

        return x

class Discriminator_spa(nn.Module):
    def __init__(self, args):
        super(Discriminator_spa, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 2, 1)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(16, 32, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.lrelu2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.lrelu3 = nn.LeakyReLU(0.2)

        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.lrelu4 = nn.LeakyReLU(0.2)

        self.conv5 = nn.Conv2d(128, 1, 4, 4, 1)
        self.lrelu5 = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.bn2(self.conv2(x)))
        x = self.lrelu3(self.bn3(self.conv3(x)))
        x = self.lrelu4(self.bn4(self.conv4(x)))
        x = self.lrelu5(self.conv5(x))

        return x