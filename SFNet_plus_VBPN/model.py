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

def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

class DnCNN(nn.Module):
    def __init__(self, in_channels, out_channels, dep=5, num_filters=64, noise_avg=False):
        '''
        Reference:
        K. Zhang, W. Zuo, Y. Chen, D. Meng and L. Zhang, "Beyond a Gaussian Denoiser: Residual
        Learning of Deep CNN for Image Denoising," TIP, 2017.

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            dep (int): depth of the network, Default 8
            num_filters (int): number of filters in each layer, Default 64
        '''
        super(DnCNN, self).__init__()
        self.conv1 = conv3x3(in_channels, num_filters, bias=True)
        self.relu = nn.LeakyReLU(0.25, True)
        mid_layer = []
        for ii in range(1, dep-1):
            mid_layer.append(conv3x3(num_filters, num_filters, bias=True))
            mid_layer.append(nn.LeakyReLU(0.25, True))
        self.mid_layer = nn.Sequential(*mid_layer)
        self.conv_last = conv3x3(num_filters, out_channels, bias=True)
        if noise_avg:
            self.global_avg = nn.AdaptiveAvgPool2d((1,1))
        else:
            self.global_avg = nn.Identity()

        self._initialize()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.mid_layer(x)
        x = self.conv_last(x)
        out = self.global_avg(x)

        return out

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.25)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

class DnCNN2(nn.Module):
    def __init__(self, in_channels, out_channels, dep=5, num_filters=64, noise_avg=False):
        '''
        Reference:
        K. Zhang, W. Zuo, Y. Chen, D. Meng and L. Zhang, "Beyond a Gaussian Denoiser: Residual
        Learning of Deep CNN for Image Denoising," TIP, 2017.

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            dep (int): depth of the network, Default 8
            num_filters (int): number of filters in each layer, Default 64
        '''
        super(DnCNN2, self).__init__()
        self.conv1 = conv3x3(in_channels, num_filters, bias=True)
        self.relu = nn.LeakyReLU(0.25, True)
        mid_layer = []
        for ii in range(1, dep-1):
            mid_layer.append(conv3x3(num_filters, num_filters, bias=True))
            mid_layer.append(nn.LeakyReLU(0.25, True))
        self.mid_layer = nn.Sequential(*mid_layer)
        self.conv_last = conv3x3(num_filters, out_channels, bias=True)
        if noise_avg:
            self.global_avg = nn.AdaptiveAvgPool2d((1,1))
        else:
            self.global_avg = nn.Identity()

        self._initialize()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.mid_layer(x)
        x = self.conv_last(x)
        out = self.global_avg(x)

        return out

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.25)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

class CALayer(nn.Module):
    def __init__(self, nf, reduction=16):
        super(CALayer, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf // reduction, 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf // reduction, nf, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg(x)
        y = self.body(y)
        return torch.mul(x, y)

class RB_Layer(nn.Module):
    def __init__(self, nf):
        super(RB_Layer, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(nf, nf, 3, 1, 1),
                                  nn.LeakyReLU(0.2, True),
                                  nn.Conv2d(nf, nf, 3, 1, 1),
                                  CALayer(nf))

    def forward(self, x):
        out = self.body(x) + x
        return out

class KernelNet(nn.Module):
    def __init__(self, in_nc=3, out_chn=3, nf=64, num_blocks=8, scale=4):
        super(KernelNet, self).__init__()

        self.head = nn.Conv2d(in_nc, nf, kernel_size=9, stride=4, padding=4, bias=False)

        self.body = nn.Sequential(*[RB_Layer(nf) for ii in range(num_blocks)])

        self.tail = nn.Sequential(nn.Conv2d(nf, out_chn, 3, 1, 1),
                                  nn.AdaptiveAvgPool2d((1, 1)))

    def forward(self, x):
        x_head = self.head(x)
        x_body = self.body(x_head)
        out = self.tail(x_body)                     # N x 3 x 1 x 1
        lam12 = torch.exp(torch.clamp(out[:, :2,], min=math.log(1e-4), max=math.log(1e2))) # N x 2 x 1 x 1
        rho = torch.tanh(out[:, -1, ]).unsqueeze(1) # N x 1 x 1 x 1
        Lam = torch.cat((lam12, rho), dim=1)        # N x 3 x 1 x 1
        return Lam
    
class AttLayer(nn.Module):
    def __init__(self, out_chn=64, extra_chn=4):
        super(AttLayer, self).__init__()

        nf1 = out_chn // 8
        nf2 = out_chn // 4

        self.conv1 = nn.Conv2d(extra_chn, nf1, kernel_size=1, stride=1, padding=0)
        self.leaky1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(nf1, nf2, kernel_size=1, stride=1, padding=0)
        self.leaky2 = nn.LeakyReLU(0.2)
        self.mul_conv = nn.Conv2d(nf2, out_chn, kernel_size=1, stride=1, padding=0)
        self.sig = nn.Sigmoid()

        self.add_conv = nn.Conv2d(nf2, out_chn, kernel_size=1, stride=1, padding=0)

    def forward(self, extra_maps):
        fea1= self.leaky1(self.conv1(extra_maps))
        fea2= self.leaky2(self.conv2(fea1))
        mul = self.sig(self.mul_conv(fea2))
        add = self.add_conv(fea2)
        return mul, add

class AttResBlock(nn.Module):
    def __init__(self, nf=64, extra_chn=4):
        super(AttResBlock, self).__init__()
        self.extra_chn = extra_chn
        if extra_chn > 0:
            self.sft1 = AttLayer(nf, extra_chn)
            self.sft2 = AttLayer(nf, extra_chn)

        self.lrelu1 = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True)

        self.lrelu2 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, feature_maps, extra_maps):
        '''
        Input:
            feature_maps: N x c x h x w
            extra_maps: N x c x h x w or None
        '''
        mul1, add1 = self.sft1(extra_maps) if self.extra_chn > 0 else (1, 0)
        fea1 = self.conv1(self.lrelu1(feature_maps * mul1 + add1))

        mul2, add2 = self.sft2(extra_maps) if self.extra_chn > 0 else (1, 0)
        fea2 = self.conv2(self.lrelu2(fea1 * mul2 + add2))
        out = torch.add(feature_maps, fea2)
        return out

class DownBlock(nn.Module):
    def __init__(self, in_chn=64, out_chn=128, extra_chn=4, n_resblocks=1, downsample=True):
        super(DownBlock, self).__init__()
        self.body = nn.ModuleList([AttResBlock(in_chn, extra_chn) for ii in range(n_resblocks)])
        if downsample:
            self.downsampler = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=2, padding=1)
        else:
            self.downsampler = nn.Identity()

    def forward(self, x, extra_maps):
        for op in self.body:
            x= op(x, extra_maps)
        out =self.downsampler(x)
        return out, x

class UpBlock(nn.Module):
    def __init__(self, in_chn=128, out_chn=64, n_resblocks=1):
        super(UpBlock, self).__init__()
        self.upsampler = nn.ConvTranspose2d(in_chn, out_chn, kernel_size=2, stride=2, padding=0)
        self.body = nn.ModuleList([AttResBlock(nf=out_chn, extra_chn=0) for _ in range(n_resblocks)])

    def forward(self, x, bridge):
        x_up = self.upsampler(x)
        for ii, op in enumerate(self.body):
            if ii == 0:
                x_up = op(x_up+bridge, None)
            else:
                x_up = op(x_up, None)
        return x_up

def pad_input(x, mod):
    h, w = x.shape[-2:]
    bottom = int(math.ceil(h/mod)*mod -h)
    right = int(math.ceil(w/mod)*mod - w)
    x_pad = torch.nn.functional.pad(x, pad=(0, right, 0, bottom), mode='reflect')
    return x_pad

class AttResUNet(nn.Module):
    def __init__(self, in_chn=3,
                 extra_chn=4,
                 out_chn=3,
                 n_resblocks=2,
                 n_feat=[64, 128, 196, 256],
                 extra_mode='Input'):
        """
        Args:
            in_chn: number of input channels
            extra_chn: number of other channels, e.g., noise variance, kernel information
            out_chn: number of output channels.
            n_resblocks: number of resblocks in each scale of UNet
            n_feat: number of channels in each scale of UNet
            extra_mode: Null, Input, Down or Both
        """
        super(AttResUNet, self).__init__()

        assert isinstance(n_feat, tuple) or isinstance(n_feat, list)
        self.depth = len(n_feat)

        self.extra_mode = extra_mode.lower()
        assert self.extra_mode in ['null', 'input', 'down', 'both']

        if self.extra_mode in ['down', 'null']:
            self.head = nn.Conv2d(in_chn, n_feat[0], kernel_size=3, stride=1, padding=1)
        else:
            self.head = nn.Conv2d(in_chn+extra_chn, n_feat[0], kernel_size=3, stride=1, padding=1)

        extra_chn_down = extra_chn if self.extra_mode.lower() in ['down', 'both'] else 0
        self.down_path = nn.ModuleList()
        for ii in range(self.depth):
            if ii+1 < self.depth:
                self.down_path.append(DownBlock(n_feat[ii], n_feat[ii+1],
                                                extra_chn=extra_chn_down,
                                                n_resblocks=n_resblocks,
                                                downsample=True))
            else:
                self.down_path.append(DownBlock(n_feat[ii], n_feat[ii],
                                      extra_chn=extra_chn_down,
                                      n_resblocks=n_resblocks,
                                      downsample=False))

        self.up_path = nn.ModuleList()
        for jj in reversed(range(self.depth - 1)):
            self.up_path.append(UpBlock(n_feat[jj+1], n_feat[jj], n_resblocks))

        self.tail = nn.Conv2d(n_feat[0], out_chn, kernel_size=3, stride=1, padding=1)

    def forward(self, x_in, extra_maps_in):
        '''
        Input:
            x_in: N x [] x h x w
            extra_maps: N x []
        '''
        h, w = x_in.shape[-2:]
        x = pad_input(x_in, 2**(self.depth-1))
        if not self.extra_mode == 'null':
            extra_maps = pad_input(extra_maps_in, 2**(self.depth-1))

        if self.extra_mode in ['input', 'both']:
            x = self.head(torch.cat([x, extra_maps], 1))
        else:
            x = self.head(x)

        blocks = []
        if self.extra_mode in ['down', 'both']:
            extra_maps_down = [extra_maps,]
        for ii, down in enumerate(self.down_path):
            if self.extra_mode in ['down', 'both']:
                x, before_down = down(x, extra_maps_down[ii])
            else:
                x, before_down = down(x, None)
            if ii != len(self.down_path)-1:
                blocks.append(before_down)
                if self.extra_mode in ['down', 'both']:
                    extra_maps_down.append(torch.nn.functional.interpolate(extra_maps, x.shape[-2:], mode='nearest'))

        for jj, up in enumerate(self.up_path):
            x = up(x, blocks[-jj-1])

        # out = self.tail(x)[..., :h, :w] + x_in
        out = self.tail(x)[..., :h, :w]

        return out

class VIRAttResUNetSR(nn.Module):
    '''
    For Denoising task with UNet denoiser.
    '''
    def __init__(self, im_chn = 16, 
                 pan_chn = 1,
                 sigma_chn=1,
                 kernel_chn=3,
                 F_chn=2,
                 n_feat=[64, 128, 192],
                 dep_S=5,
                 dep_K=8,
                 noise_cond=False,
                 kernel_cond=False,
                 F_cond=False,
                 n_resblocks=1,
                 extra_mode='null',
                 noise_avg=False):
        super(VIRAttResUNetSR, self).__init__()
        self.noise_cond = noise_cond
        self.noise_avg = noise_avg
        self.kernel_cond = kernel_cond
        self.F_cond = F_cond

        extra_chn = 0
        if self.kernel_cond: extra_chn += kernel_chn
        if self.noise_cond: extra_chn += sigma_chn
        if self.F_cond: extra_chn += F_chn
        self.SNet = DnCNN(im_chn, sigma_chn, dep=dep_S, noise_avg=noise_avg)
        self.FNet = DnCNN2(pan_chn, pan_chn*2, dep=dep_S*2, noise_avg=noise_avg)
        self.KNet = KernelNet(im_chn, kernel_chn, num_blocks=dep_K)
        in_ch = im_chn + pan_chn
        self.RNet = AttResUNet(in_ch,
                               extra_chn=extra_chn,
                               out_chn=im_chn,
                               n_feat=n_feat,
                               n_resblocks=n_resblocks,
                               extra_mode=extra_mode)

    def forward(self, x, p, sf):
        sigma = torch.exp(torch.clamp(self.SNet(x), min=math.log(1e-10), max=math.log(1e2)))  # N x [] x 1 x 1
        kinfo_est = self.KNet(x)    # N x [] x 1 x 1
        alpha = self.FNet(p)
        x_up = torch.nn.functional.interpolate(x, scale_factor=sf, mode='nearest')
        h_up, w_up = x_up.shape[-2:]
        if not self.noise_cond and not self.kernel_cond and not self.F_cond:
            extra_maps = None
        else:
            extra_temp = []
            if self.kernel_cond: extra_temp.append(kinfo_est.repeat(1,1,h_up,w_up))
            if self.noise_cond:
                if self.noise_avg:
                    extra_temp.append(sigma.sqrt().repeat(1,1,h_up,w_up))
                else:
                    extra_temp.append(torch.nn.functional.interpolate(sigma.sqrt(), scale_factor=sf, mode='nearest'))
            if self.F_cond:
                if self.noise_avg:
                    extra_temp.append(alpha.repeat(1,1,h_up,w_up))
                else:
                    extra_temp.append(alpha)
            extra_maps = torch.cat(extra_temp, 1)    # n x [] x h x w

        # added by lhx 2023-12-13
        # concat input features
        x_in = [x_up, p]
        x_in = torch.cat(x_in, 1)
        mu = self.RNet(x_in, extra_maps)
        # end added
        return mu, kinfo_est.squeeze(-1).squeeze(-1), sigma, alpha

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train')
    args = parser.parse_args()

    args.num_bands = 16

    sfnet = SFNet(args)
    sfnet = sfnet.to("cuda:4")

    a = torch.randn(4, 1, 128, 128).to("cuda:4")
    b = sfnet(a)
    torch.save(sfnet.state_dict(), "net.pth")

