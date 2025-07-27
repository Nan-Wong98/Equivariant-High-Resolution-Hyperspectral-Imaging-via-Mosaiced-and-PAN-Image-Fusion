import torch
import torch.nn as nn
import math
import numpy as np
import utils
from neighborhood_attention import NATLayer  # Neighbourhood Attention Transformer
from neighborhood_attention import Channel_Layernorm

class FFN(nn.Module): # Feed Forward Network
    def __init__(self, dim):
        super(FFN, self).__init__()
        self.spatial_convs = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1),
                                            nn.Conv2d(dim, dim, 3, 1, 1),
                                            nn.Conv2d(dim, dim, 3, 1, 1),
                                            )
        self.frequency_real_convs = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1),
                                                    nn.Conv2d(dim, dim, 3, 1, 1),
                                                    nn.Conv2d(dim, dim, 3, 1, 1),
                                                    )
        self.frequency_image_convs = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1),
                                                    nn.Conv2d(dim, dim, 3, 1, 1),
                                                    nn.Conv2d(dim, dim, 3, 1, 1),
                                                    )
    
    def forward(self, x):
        spatial_branch = self.spatial_convs(x)
        x_fft = torch.fft.fft2(x)
        real = torch.abs(x_fft)
        image = torch.angle(x_fft)
        real = self.frequency_real_convs(real)
        image = self.frequency_image_convs(image)
        frequency_branch = torch.fft.ifft2(real*torch.exp(1j*image)).real
        identity_branch = x

        return spatial_branch + frequency_branch + identity_branch

class FAF(nn.Module): # Frequency Adaptive Filtering
    def __init__(self, dim):
        super(FAF, self).__init__()
        self.conv_gamma_real = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv_theta_real = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv_gamma_image = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv_theta_image = nn.Conv2d(dim, dim, 3, 1, 1)
    
    def forward(self, x):
        gamma_real = self.conv_gamma_real(x)
        theta_real = self.conv_theta_real(x)
        gamma_image = self.conv_gamma_image(x)
        theta_image = self.conv_theta_image(x)
        x_fft = torch.fft.fft2(x)
        x_fft_real = torch.abs(x_fft)
        x_fft_image = torch.angle(x_fft)

        y_real = x_fft_real * gamma_real + theta_real
        y_image = x_fft_image * gamma_image + gamma_image
        y_ifft = torch.fft.ifft2(y_real*torch.exp(1j*y_image)).real
        return y_ifft

class SFTL(nn.Module): # Spatial Frequency Transformer Layer
    def __init__(self, dim):
        super(SFTL, self).__init__()
        self.layernorm1 = Channel_Layernorm(dim)
        self.faf = FAF(dim//2)
        self.sna = NATLayer(dim//2, num_heads=4)
        self.layernorm2 = Channel_Layernorm(dim)
        self.ffn = FFN(dim)
    
    def forward(self, x):
        shortcut = x
        x = self.layernorm1(x)
        c = x.shape[1]
        x_faf = self.faf(x[:, :c//2])
        x_sna = self.sna(x[:, c//2:])
        x = self.layernorm2(torch.cat((x_faf, x_sna), 1))
        x = self.ffn(x)
        return x + shortcut

class RSFTB(nn.Module): # Residual Spatial-frequency Transformer Block
    def __init__(self, dim):
        super(RSFTB, self).__init__()
        self.SFTLs = nn.Sequential(SFTL(dim),
                                    SFTL(dim),
                                    SFTL(dim),
                                    SFTL(dim),
                                    )
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
    
    def forward(self, x):
        shortcut = x
        x = self.conv(self.SFTLs(x))
        return x + shortcut

def FT_init(mosaic):
    mosaic_fft = torch.fft.fft2(mosaic)
    amplitude = torch.abs(mosaic_fft)
    phase = torch.angle(mosaic_fft)
    amplitude_repeat = amplitude.repeat(mosaic.shape[1], dim=1)
    phase_repeat = phase.repeat_interleave(mosaic.shape[1], dim=1)
    demosaic = torch.fft.ifft2(amplitude_repeat * torch.exp(1j * phase_repeat)).real
    demosaic = torch.nn.functional.pixel_shuffle(demosaic, 4)
    return demosaic
    
class SFNet(nn.Module): # FT-SFNet
    def __init__(self, args):
        super(SFNet, self).__init__()
        self.dim = 48
        self.RSFTBs = nn.Sequential(RSFTB(self.dim),
                                    RSFTB(self.dim),
                                    RSFTB(self.dim),
                                    RSFTB(self.dim),
                                    RSFTB(self.dim),
                                    )
        self.conv1 = nn.Conv2d(args.num_bands, self.dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(self.dim*2, args.num_bands, 3, 1, 1)
    
    def forward(self, x):
        x = torch.nn.functional.pixel_unshuffle(x, 4)
        x_demosaic_init = torch.nn.functional.interpolate(x, scale_factor=4, mode="bicubic")
        # x_demosaic_init = FT_init(x)
        x = self.conv1(x_demosaic_init)
        x = self.conv2(torch.cat((self.RSFTBs(x), x), 1))
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