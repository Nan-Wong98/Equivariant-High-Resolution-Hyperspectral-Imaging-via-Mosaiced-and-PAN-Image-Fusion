import torch
import torch.nn as nn
import math
import numpy as np
import utils

class L1_Charbonnier_mean_loss_for_mosaic(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self, MSFA, device):
        super(L1_Charbonnier_mean_loss_for_mosaic, self).__init__()
        self.eps = 1e-6
        self.msfa_kernel = torch.zeros(MSFA.shape[0] * MSFA.shape[1], 1, MSFA.shape[0], MSFA.shape[1]).to(device)
        for i in range(MSFA.shape[0]):
            for j in range(MSFA.shape[1]):
                self.msfa_kernel[int(MSFA[i, j]), 0, i, j] += 1

    def forward(self, X, Y):
        assert X.shape[1] == self.msfa_kernel.shape[0] * self.msfa_kernel.shape[1]
        X_Mosaic = torch.nn.functional.conv2d(X, self.msfa_kernel, bias=None, stride=self.msfa_kernel.shape[2], groups=X.shape[1])
        X_Mosaic = torch.nn.functional.pixel_shuffle(X_Mosaic, upscale_factor=self.msfa_kernel.shape[2])
        diff = torch.add(X_Mosaic, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

class L1_Charbonnier_mean_loss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_mean_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

class reconstruction_loss(nn.Module):
    """reconstruction loss of raw_msfa"""

    def __init__(self, msfa_size):
        super(reconstruction_loss, self).__init__()
        self.wt = 1
        self.msfa_size = msfa_size
        # self.mse_loss = nn.MSELoss(reduce=True, size_average=False)
        self.mse_loss = L1_Charbonnier_mean_loss()

    def get_msfa(self, img_tensor, msfa_size):
        mask = torch.zeros_like(img_tensor)
        for i in range(0, msfa_size):
            for j in range(0, msfa_size):
                mask[:, i * msfa_size + j, i::msfa_size, j::msfa_size] = 1
        # buff_raw1 = mask[0, 1, :, :].cpu().detach().numpy()
        # buff_raw2 = img_tensor[0, 1, :, :].cpu().detach().numpy()
        return torch.sum(mask.mul(img_tensor), 1)

    def forward(self, X, Y):
        # loss = self.mse_loss(self.get_msfa(X, self.msfa_size), self.get_msfa(Y, self.msfa_size))
        loss = self.mse_loss(X, Y)
        return loss

class Pos2Weight(nn.Module):
    def __init__(self, outC=16, kernel_size=5, inC=1):
        super(Pos2Weight, self).__init__()
        self.inC = inC
        self.kernel_size = kernel_size
        self.outC = outC
        self.meta_block = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.kernel_size * self.kernel_size * self.inC * self.outC)
        )

    def forward(self, x):
        output = self.meta_block(x)
        return output

class ChannelPool(nn.Module):
    def forward(self, x):
        # return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
        # return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
        # return torch.mean(x,1)
        return torch.mean(x, 1).unsqueeze(1)

class Shuffle_d(nn.Module):
    def __init__(self, scale=2):
        super(Shuffle_d, self).__init__()
        self.scale = scale

    def forward(self, x):
        def _space_to_channel(x, scale):
            b, C, h, w = x.size()
            Cout = C * scale ** 2
            hout = h // scale
            wout = w // scale
            x = x.contiguous().view(b, C, hout, scale, wout, scale)
            x = x.contiguous().permute(0, 1, 3, 5, 2, 4)
            x = x.contiguous().view(b, Cout, hout, wout)
            return x
        return _space_to_channel(x, self.scale)
    
class CA_AA_par_Layer1(nn.Module):
    def __init__(self, msfa_size, channel, reduction=16):
        super(CA_AA_par_Layer1, self).__init__()
        self.compress = ChannelPool()
        self.shuffledown = Shuffle_d(msfa_size)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(msfa_size**2, msfa_size**2 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(msfa_size**2 // reduction, msfa_size**2, bias=False),
            nn.Sigmoid()
        )
        self.shuffleup = nn.PixelShuffle(msfa_size)

        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        buff_x = x
        N, C, H, W = x.size()
        x = x.view(N * C, 1, H, W)  # N, C, H, W to N*C, 1, H, W
        sq_x = self.shuffledown(x)  # N*C, 1, H, W to N*C, 16, H/4, W/4
        b, c, _, _ = sq_x.size()
        y = self.avg_pool(sq_x).view(b, c)  # N*C, 16, H/4, W/4 to N*C, 16, 1, 1 to N*C, 16
        y = self.fc(y).view(b, c, 1, 1)  # N*C, 16, 1, 1
        y = y.expand_as(sq_x)  # N*C, 16, 1, 1 to N*C, 16, H/4, W/4
        ex_y = self.shuffleup(y)  # N*C, 16, H/4, W/4 to N*C, 1, H, W
        out = x * ex_y
        out = out.view(N, C, H, W)

        b, c, _, _ = buff_x.size()
        y = self.avg_pool1(buff_x).view(b, c)
        y = self.fc1(y).view(b, c, 1, 1)
        out = out * y.expand_as(out)
        return out

class _Conv_LSA_Block_msfasize(nn.Module):
    def __init__(self, msfa_size):
        super(_Conv_LSA_Block_msfasize, self).__init__()
        self.ma = CA_AA_par_Layer1(msfa_size, 64, 4)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.cov_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
        )

    def forward(self, x):
        residual = x
        output = self.cov_block(x)
        output = self.ma(output)
        output += residual
        output = self.relu(output)
        return output

def get_WB_filter_msfa(msfa_size):
    """make a 2D weight bilinear kernel suitable for WB_Conv"""
    size = 2*msfa_size-1
    ligne = []
    colonne = []
    for i in range(size):
        if (i + 1) <= np.floor(math.sqrt(msfa_size**2)):
            ligne.append(i + 1)
            colonne.append(i + 1)
        else:
            ligne.append(ligne[i - 1] - 1.0)
            colonne.append(colonne[i - 1] - 1.0)
    BilinearFilter = np.zeros(size * size)
    for i in range(size):
        for j in range(size):
            BilinearFilter[(j + i * size)] = (ligne[i] * colonne[j] / (msfa_size**2))
    filter0 = np.reshape(BilinearFilter, (size, size))
    return torch.from_numpy(filter0).float()

class Mpattern_opt(nn.Module):
    def __init__(self, args):
        super(Mpattern_opt, self).__init__()
        self.scale = 1
        msfa_size = args.msfa_size
        self.msfa_size = args.msfa_size
        self.outC = args.msfa_size**2
        if msfa_size == 5:
            self.mcm_ksize = msfa_size+2
        elif msfa_size == 4:
            self.mcm_ksize = msfa_size + 1
        self.WB_Conv = nn.Conv2d(in_channels=msfa_size**2, out_channels=msfa_size**2, kernel_size=2*msfa_size-1, stride=1, padding=msfa_size-1, bias=False, groups=msfa_size**2)
        self.P2W = Pos2Weight(outC=self.outC, kernel_size=self.mcm_ksize)
        self.att = CA_AA_par_Layer1(msfa_size, msfa_size ** 2, 4)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv_input = nn.Conv2d(in_channels=msfa_size**2, out_channels=64, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        self.convt_F1 = self.make_ma_layer(_Conv_LSA_Block_msfasize, msfa_size)
        self.convt_F2 = self.make_ma_layer(_Conv_LSA_Block_msfasize, msfa_size)

        self.conv_tail = nn.Conv2d(in_channels=64, out_channels=msfa_size ** 2, kernel_size=3, stride=1, padding=1, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.groups == msfa_size ** 2:
                    c1, c2, h, w = m.weight.data.size()
                    WB = get_WB_filter_msfa(msfa_size)
                    for i in m.parameters():
                        i.requires_grad = False
                    m.weight.data = WB.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def make_ma_layer(self, block, msfa_size):
        layers = []
        layers.append(block(msfa_size))
        return nn.Sequential(*layers)

    def forward_once(self, x):
        x = self.conv_input(x)
        out = self.convt_F1(x)
        out = self.convt_F2(out)
        return out

    def repeat_y(self, y):
        scale_int = math.ceil(self.scale)
        N, C, H, W = y.size()
        y = y.view(N, C, H, 1, W, 1)

        y = torch.cat([y] * scale_int, 3)
        y = torch.cat([y] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)

        return y.contiguous().view(-1, C, H, W)

    def forward(self, data, pos_mat):
        x, y = data
        WB_norelu = self.WB_Conv(x)
        N, C, H, W = y.size()
        # print(H, W)
        pos_mat = pos_mat.view(1, H, W, 2)
        pos_mat = pos_mat[:, 0:self.msfa_size, 0:self.msfa_size, :]
        pos_mat = pos_mat.contiguous().view(1, self.msfa_size ** 2, 2)
        local_weight = self.P2W(pos_mat.view(pos_mat.size(1), -1))
        local_weight = local_weight.view(self.msfa_size, self.msfa_size, self.outC * self.mcm_ksize * self.mcm_ksize)
        local_weight1 = local_weight.clone()
        cols = nn.functional.unfold(y, self.mcm_ksize, padding=(self.mcm_ksize - 1) // 2)
        cols = cols.contiguous().view(cols.size(0), 1, cols.size(1), cols.size(2),
                                      1).permute(0, 1, 3, 4, 2).contiguous()

        h_pattern_n = 1
        # This h_pattern_n can divide H / msfa_size as a int
        local_weight1 = local_weight1.repeat(h_pattern_n, int(W / self.msfa_size), 1)
        # print(local_weight1.size())
        # print(h_pattern_n, self.msfa_size, W)
        local_weight1 = local_weight1.view(h_pattern_n * self.msfa_size * W, self.outC * self.mcm_ksize * self.mcm_ksize)
        local_weight1 = local_weight1.contiguous().view(1, h_pattern_n * self.msfa_size * W, -1, self.outC)
        for i in range(0, int(H / self.msfa_size / h_pattern_n)):
            cols_buff = cols[:, 0, i * self.msfa_size * h_pattern_n * W:(i + 1) * self.msfa_size * h_pattern_n * W, :, :]
            if i == 0:
                Raw_conv_buff = torch.matmul(cols_buff, local_weight1)
            else:
                Raw_conv_buff = torch.cat([Raw_conv_buff, torch.matmul(cols_buff, local_weight1)], dim=-3)

        Raw_conv_buff = torch.unsqueeze(Raw_conv_buff, 0)
        Raw_conv_buff = Raw_conv_buff.permute(0, 1, 4, 2, 3)
        Raw_conv_buff = Raw_conv_buff.contiguous().view(N, 1, 1, self.outC, H, W)
        Raw_conv_buff = Raw_conv_buff.contiguous().view(N, self.outC, H, W)

        out = self.att(Raw_conv_buff)
        out = self.relu1(out)
        out = self.forward_once(out)
        out = self.conv_tail(out)
        return torch.add(out, WB_norelu)
        # return WB_norelu


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