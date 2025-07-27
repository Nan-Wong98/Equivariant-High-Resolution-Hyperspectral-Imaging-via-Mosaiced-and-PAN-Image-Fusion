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
import time
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
        # t = time.time()
        for i in range(0, int(H / self.msfa_size / h_pattern_n)):
            cols_buff = cols[:, 0, i * self.msfa_size * h_pattern_n * W:(i + 1) * self.msfa_size * h_pattern_n * W, :, :]
            if i == 0:
                Raw_conv_buff = torch.matmul(cols_buff, local_weight1)
            else:
                Raw_conv_buff = torch.cat([Raw_conv_buff, torch.matmul(cols_buff, local_weight1)], dim=-3)
        # print(time.time() - t)

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