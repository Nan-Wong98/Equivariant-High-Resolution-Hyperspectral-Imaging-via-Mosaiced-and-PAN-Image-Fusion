import time
import os
import torch
from torch.utils.data import DataLoader
from GetDataSet import MakeRealWorldDatasetForPansharpening
import torch.nn as nn
from model import Mpattern_opt, VIRAttResUNetSR
import copy
import argparse
import json
import cv2, math
import numpy
import utils
import random
import tqdm
import quality_index

def cal_kl_gauss_simple(mu_q, mu_p, var_p): return 0.5 * ((mu_q-mu_p)**2/var_p).mean()

def cal_kl_inverse_gamma_simple(beta_q, alpha_p, beta_p):
    out = alpha_p*(beta_p.div(beta_q)-1) + alpha_p*(beta_q.log()-beta_p.log())
    return out.mean()

def reparameter_inv_gamma(alpha, beta):
    dist_gamma = torch.distributions.gamma.Gamma(alpha, beta)
    out = 1 / dist_gamma.rsample()
    return out

def reparameter_cov_mat(kinfo_est, kappa0, rho_var):
    '''
    Reparameterize kernelo.
    Input:
        kinfo_est: N x 3
    '''
    alpha_k = torch.ones_like(kinfo_est[:, :2]) * (kappa0-1)
    beta_k = kinfo_est[:, :2] * kappa0
    k_var = reparameter_inv_gamma(alpha_k, beta_k)
    k_var1, k_var2 = torch.chunk(k_var, 2, dim=1)    # N x 1, resampled variance along x and y axis
    rho_mean = kinfo_est[:, 2].unsqueeze(1)          # N x 1, mean of the correlation coffecient
    rho = rho_mean + math.sqrt(rho_var)*torch.randn_like(rho_mean)  # resampled correlation coffecient
    direction = k_var1.detach().sqrt() * k_var2.detach().sqrt() * torch.clamp(rho, min=-1, max=1)   # N x 1
    k_cov = torch.cat([k_var1, direction, direction, k_var2], dim=1).view(-1, 1, 2, 2) # N x 1 x 2 x 2
    return k_cov

def sigma2kernel(sigma, k_size=21, sf=3, shift=False):
    '''
    Generate Gaussian kernel according to cholesky decomposion.
    Input:
        sigma: N x 1 x 2 x 2 torch tensor, covariance matrix
        k_size: integer, kernel size
        sf: scale factor
    Output:
        kernel: N x 1 x k x k torch tensor
    '''
    try:
        sigma_inv = torch.inverse(sigma)
    except:
        sigma_disturb = sigma + torch.eye(2, dtype=sigma.dtype, device=sigma.device).unsqueeze(0).unsqueeze(0) * 1e-5
        sigma_inv = torch.inverse(sigma_disturb)

    # Set expectation position (shifting kernel for aligned image)
    if shift:
        center = k_size // 2 + 0.5 * (sf - k_size % 2)                         # + 0.5 * (sf - k_size % 2)
    else:
        center = k_size // 2

    # Create meshgrid for Gaussian
    X, Y = torch.meshgrid(torch.arange(k_size), torch.arange(k_size))
    Z = torch.stack((X, Y), dim=2).to(device=sigma.device, dtype=sigma.dtype).view(1, -1, 2, 1)      # 1 x k^2 x 2 x 1

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z - center                                                        # 1 x k^2 x 2 x 1
    ZZ_t = ZZ.permute(0, 1, 3, 2)                                          # 1 x k^2 x 1 x 2
    ZZZ = -0.5 * ZZ_t.matmul(sigma_inv).matmul(ZZ).squeeze(-1).squeeze(-1) # N x k^2
    kernel = torch.nn.functional.softmax(ZZZ, dim=1)                                         # N x k^2

    return kernel.view(-1, 1, k_size, k_size)                # N x 1 x k x k

def conv_multi_kernel_tensor(im_hr, kernel, sf):
    '''
    Degradation model by Pytorch.
    Input:
        im_hr: N x c x h x w
        kernel: N x 1 x k x k
        sf: scale factor
    '''
    im_hr_pad = torch.nn.functional.pad(im_hr, (kernel.shape[-1] // 2,)*4, mode='reflect')
    im_blur = torch.nn.functional.conv3d(im_hr_pad.unsqueeze(0), kernel.unsqueeze(1), groups=im_hr.shape[0])
    im_blur = torch.nn.functional.interpolate(im_blur[0], scale_factor=1/sf, mode="bicubic")

    return im_blur

def cal_likelihood_sisr(x, kernel, sf, mu_q, var_q, alpha_q, beta_q, downsampler):
    zz = mu_q + torch.randn_like(mu_q) * math.sqrt(var_q)
    zz_blur = conv_multi_kernel_tensor(zz, kernel, sf)
    out = 0.5*math.log(2*math.pi) +  0.5*(beta_q.log()-alpha_q.digamma()) +  0.5*alpha_q.div(beta_q)*(x-zz_blur)**2
    return out.mean()

def criterion(mu, sigma_est, kinfo_est, alpha_est, 
            im_hr,
            ms_lr,
            sigma_prior,
            ms_alpha0,
            kinfo_gt,
            kappa0,
            r2,
            eps2,
            sf,
            k_size,
            penalty_K,
            shift,
            downsampler,
            # PAN部分
            pan_n,
            pan_label,
            pan_sigmaMap,
            pan_eps2,
            device):

        '''
        MS部分loss
        '''
        # KL divergence for Gauss distribution
        if isinstance(mu, list):
            kl_rnet = cal_kl_gauss_simple(mu[0], im_hr, eps2)
            for jj in range(1, len(mu)):
                kl_rnet += cal_kl_gauss_simple(mu[jj], im_hr, eps2)
            kl_rnet /= len(mu)
        else:
            kl_rnet = cal_kl_gauss_simple(mu, im_hr, eps2)

        # KL divergence for Inv-Gamma distribution of the sigma map for noise
        beta0 = sigma_prior * ms_alpha0
        beta = sigma_est * ms_alpha0
        kl_snet = cal_kl_inverse_gamma_simple(beta, ms_alpha0-1, beta0)

        # KL divergence for the kernel
        kl_knet0 = cal_kl_inverse_gamma_simple(kappa0*kinfo_est[:, 0], kappa0-1, kappa0*kinfo_gt[:, 0])
        kl_knet1 = cal_kl_inverse_gamma_simple(kappa0*kinfo_est[:, 1], kappa0-1, kappa0*kinfo_gt[:, 1])
        kl_knet2 = cal_kl_gauss_simple(kinfo_est[:, 2], kinfo_gt[:, 2], r2) * penalty_K[0]
        kl_knet = (kl_knet0 + kl_knet1 + kl_knet2) / 3 * penalty_K[1]

        # reparameter kernel
        k_cov = reparameter_cov_mat(kinfo_est, kappa0, r2)        # resampled covariance matrix, N x 1 x 2 x 2
        kernel = sigma2kernel(k_cov, k_size, sf, shift)        # N x 1 x k x k

        # likelihood
        if isinstance(mu, list):
            lh = cal_likelihood_sisr(ms_lr, kernel, sf, mu[0], eps2, ms_alpha0-1, beta, downsampler)
            for jj in range(1, len(mu)):
                lh += cal_likelihood_sisr(ms_lr, kernel, sf, mu[jj], eps2, ms_alpha0-1, beta, downsampler)
            lh /= len(mu)
        else:
            lh = cal_likelihood_sisr(ms_lr, kernel, sf, mu, eps2, ms_alpha0-1, beta, downsampler)

        loss = lh + kl_rnet + kl_snet + kl_knet

        '''
        PAN部分loss 
        '''
        radius = 3
        out_denoise = mu
        out_denoise = torch.mean(out_denoise, dim=1, keepdim=True)

        # 定义 Sobel 算子
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32)

        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32)   
        
        # 添加维度以匹配图像张量的维度
        sobel_x = sobel_x.view(1, 1, 3, 3).to(device)
        sobel_y = sobel_y.view(1, 1, 3, 3).to(device)

        # 对图像张量进行 Sobel 算子的卷积操作
        with torch.no_grad():
            out_denoise_gradient_x = torch.nn.functional.conv2d(out_denoise, sobel_x, padding=1)
            out_denoise_gradient_y = torch.nn.functional.conv2d(out_denoise, sobel_y, padding=1)

            # 计算梯度的幅值
            out_denoise_gradient_magnitude = torch.sqrt(out_denoise_gradient_x**2 + out_denoise_gradient_y**2)

            # pan_label求梯度
            pan_label_gradient_x = torch.nn.functional.conv2d(pan_label, sobel_x, padding=1)
            pan_label_gradient_y = torch.nn.functional.conv2d(pan_label, sobel_y, padding=1)

            # 计算梯度的幅值
            pan_label_gradient_magnitude = torch.sqrt(pan_label_gradient_x**2 + pan_label_gradient_y**2)

        out_sigma = alpha_est
        log_max = math.log(1e4)
        log_min = math.log(1e-8)
        C = pan_label.shape[1]
        p = 2 * radius + 1
        p2 = p ** 2
        pan_alpha0 = 0.5 * torch.tensor([p2 - 2]).type(pan_sigmaMap.dtype).to(device)
        pan_beta0 = 0.5 * p2 * pan_sigmaMap
        out_denoise_gradient_magnitude[:, C:, ].clamp_(min=log_min, max=log_max)
        err_mean = out_denoise_gradient_magnitude[:, :C, ]
        m2 = torch.exp(out_denoise_gradient_magnitude[:, C:, ])  # variance

        out_sigma.clamp_(min=log_min, max=log_max)
        log_alpha = out_sigma[:, :C, ]
        alpha = torch.exp(log_alpha)
        log_beta = out_sigma[:, C:, ]
        alpha_div_beta = torch.exp(log_alpha - log_beta)

        kl_gauss = 0.5 * torch.mean((out_denoise_gradient_magnitude - pan_label_gradient_magnitude) ** 2 / pan_eps2)

        kl_Igamma = torch.mean((alpha - pan_alpha0) * torch.digamma(alpha) + (torch.lgamma(pan_alpha0) - torch.lgamma(alpha))
                               + pan_alpha0 * (log_beta - torch.log(pan_beta0)) + pan_beta0 * alpha_div_beta - alpha)

        pan_lh = 0.5 * math.log(2 * math.pi) + 0.5 * torch.mean(
            (log_beta - torch.digamma(alpha)) + ((out_denoise_gradient_magnitude - pan_label_gradient_magnitude) ** 2 + pan_eps2) * alpha_div_beta)        
        loss = loss + pan_lh + kl_gauss + kl_Igamma

        return loss

def train(ps_net: nn.Module, optimizer, train_dataloader, val_dataloader, args):
    print('===>Begin Training!')
    start_epoch = 0
    if args.resume != "":
        start_epoch = int(args.resume) if "best" not in args.resume else int(args.resume.split("_")[-1])

    best_epoch, best_qnr = 0, 0
    MSFA = numpy.array([[0, 1, 2, 3],
                        [4, 5, 6, 7],
                        [8, 9, 10, 11],
                        [12, 13, 14, 15]])
    device = next(ps_net.parameters()).device
    msfa_kernel = torch.zeros(MSFA.shape[0] * MSFA.shape[1], 1, MSFA.shape[0]*2, MSFA.shape[1]*2).to(device)
    for i in range(MSFA.shape[0]):
        for j in range(MSFA.shape[1]):
            msfa_kernel[int(MSFA[i, j]), 0, i*2, j*2] = 0.25
            msfa_kernel[int(MSFA[i, j]), 0, i*2+1, j*2] = 0.25
            msfa_kernel[int(MSFA[i, j]), 0, i*2, j*2+1] = 0.25
            msfa_kernel[int(MSFA[i, j]), 0, i*2+1, j*2+1] = 0.25

    l1_loss = nn.L1Loss()
    t = time.time()
    
    for epoch in range(start_epoch + 1, args.epochs + 1):
        # train
        ps_net.train()
        start_time = time.time()

        for cnt, data in enumerate(tqdm.tqdm(train_dataloader)):
            ms_lable, ms_lr, ms_blur, kinfo_gt, nlevel, \
                pan_label, pan_n, pan_map, pan_eps2 = [x.to(device) for x in data]
            ms_sigma_prior = nlevel
            ms_alpha0 = 0.5 * torch.tensor([9**2], dtype=torch.float32).to(device)
            kappa0 = torch.tensor([50], dtype=torch.float32).to(device)

            optimizer.zero_grad()
            mu, kinfo_est, sigma_est, alpha_est = ps_net(ms_lr, pan_n, args.spatial_ratio)
            loss = criterion(mu=mu, 
                            kinfo_est=kinfo_est, 
                            sigma_est=sigma_est, 
                            alpha_est=alpha_est, 
                            im_hr=ms_lable, 
                            ms_lr=ms_lr,
                            sigma_prior=ms_sigma_prior,
                            ms_alpha0=ms_alpha0,
                            kinfo_gt=kinfo_gt,
                            kappa0=kappa0,
                            r2=1e-4,
                            eps2=1e-6,
                            sf=args.spatial_ratio,
                            k_size=11,
                            penalty_K=[0.02, 2],
                            downsampler="bicubic",
                            shift=False,
                            pan_n=pan_n,
                            pan_label=pan_label,
                            pan_sigmaMap=pan_map,
                            pan_eps2=pan_eps2,
                            device=device)
            loss.backward()
            optimizer.step()

        # val
        qnr_avg, d_lambda_avg, d_s_avg = 0., 0., 0.
        ps_net.eval()
        with torch.no_grad():
            for cnt, data in enumerate(val_dataloader):
                mosaic, lrms, pan = data[0].to(device), data[1].to(device),  data[2].to(device)
                upms = torch.nn.functional.interpolate(lrms, scale_factor=args.spatial_ratio, mode="bicubic")

                fused = ps_net._modules["RNet"](torch.cat((upms, pan), 1), None).detach()
                fused = torch.clamp(torch.round(fused*2**12), 0, 2**12).short()

                qnr, d_lambda, d_s = quality_index.calc_qnr_mosaic(fused.double()/2**12, mosaic.double(), pan.double(), msfa_kernel, patch_size=32, scale_factor=args.spatial_ratio)
                qnr_avg += qnr.item()
                d_lambda_avg += d_lambda.item()
                d_s_avg += d_s.item()

                if args.visual == True and epoch % args.visual_freq == 0 and epoch != 0:
                    rgb_bands = [0, 1, 13]
                    if not os.path.exists(os.path.join(args.dir_record, str(cnt))):
                        os.mkdir(os.path.join(args.dir_record, str(cnt)))
                    if not os.path.exists(os.path.join(args.dir_record, str(cnt), "mosaic.png")) or epoch == args.visual_freq:
                        mosaic_numpy = utils.pixel_shuffle_inv(mosaic.cpu().numpy(), 4)[0].transpose(1, 2, 0)[:, :, rgb_bands]
                        mosaic_numpy = numpy.clip(numpy.round(mosaic_numpy * 255), 0, 255)
                        cv2.imwrite(os.path.join(args.dir_record, str(cnt), "mosaic.png"), mosaic_numpy)
                        lrms_numpy = lrms[0].permute(1, 2, 0).cpu().numpy()[:, :, rgb_bands]
                        lrms_numpy = numpy.clip(numpy.round(lrms_numpy * 255), 0, 255)
                        cv2.imwrite(os.path.join(args.dir_record, str(cnt), "demosaic.png"), lrms_numpy)
                        pan_numpy = pan.cpu().numpy()[0].transpose(1, 2, 0)
                        pan_numpy = numpy.clip(numpy.round(pan_numpy * 255), 0, 255)
                        cv2.imwrite(os.path.join(args.dir_record, str(cnt), "pan.png"), pan_numpy)
                    
                    fused_numpy = fused[0].permute(1, 2, 0).cpu().numpy()[:, :, rgb_bands]
                    fused_numpy = numpy.clip(numpy.round(fused_numpy / 2**12 * 255), 0, 255)
                    cv2.imwrite(os.path.join(args.dir_record, str(cnt), f"fused_{epoch}.png"), fused_numpy)

        qnr_avg /= cnt+1
        d_lambda_avg /= cnt+1
        d_s_avg /= cnt+1

        # save model with highest QNR
        if qnr_avg > best_qnr:
            best_qnr = qnr_avg
            if best_epoch != 0:
                os.remove(os.path.join(args.dir_model, "best_{}.pth".format(best_epoch)))
            best_epoch = epoch
            torch.save(ps_net.state_dict(), os.path.join(args.dir_model, "best_{}.pth".format(epoch)))

        if args.record is not False:
            record = []
            if os.path.exists(args.record):
                with open(args.record, "r") as f:
                    record = json.load(f)
            record.append({"epoch": epoch, "qnr": qnr_avg, "best_qnr": best_qnr, "best_epoch": best_epoch, "learning rate": optimizer.param_groups[0]["lr"]})
            with open(args.record, "w") as f:
                record = json.dump(record, f, indent=2)

        # save model at some frequency
        if epoch % args.save_freq == 0:
            torch.save(ps_net.state_dict(), os.path.join(args.dir_model, f"{epoch}.pth"))
        
        # log
        print("Epoch: ", epoch,
            "QNR: %.4f"%qnr_avg,
            "time: %.2f"%((time.time() - start_time) / 60), "min", 
            "best_QNR: %.4f"%best_qnr,
            "best_epoch: ", best_epoch,
            "learning rate: ", optimizer.param_groups[0]["lr"]
            )

    print(f"Total time: {(time.time() - t) / 60} min")
    print("Best_epoch: {}, QNR: {}".format(best_epoch, best_qnr))


def main(args):
    dir_idx = os.path.join("./", str(args.idx))
    if not os.path.exists(dir_idx):
        os.mkdir(dir_idx)
    args.cache_path = dir_idx

    dir_model = os.path.join(dir_idx, "model")
    if not os.path.exists(dir_model):
        os.mkdir(dir_model)
    args.dir_model = dir_model

    if args.record is True:
        dir_record = os.path.join(dir_idx, "record")
        if not os.path.exists(dir_record):
            os.makedirs(dir_record)
        args.dir_record = dir_record
        args.record = os.path.join(dir_record, "record.json")
        if args.resume == "" and os.path.exists(args.record):
            os.remove(args.record)

    demosaic_net = Mpattern_opt(args)
    demosaic_net.load_state_dict(torch.load(args.resume_demosaic), strict=False)
    demosaic_net = demosaic_net.to(f"cuda:{args.device}")
    train_set = MakeRealWorldDatasetForPansharpening(args, "train", demosaic_net)
    train_dataloader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
    val_set = MakeRealWorldDatasetForPansharpening(args, "test", demosaic_net)
    val_dataloader = DataLoader(dataset=val_set, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                pin_memory=True)

    total_iterations = args.epochs * len(train_dataloader)
    print('total_iterations:{}'.format(total_iterations))

    ps_net = VIRAttResUNetSR()

    if args.resume != "":
        backup_pth = os.path.join(dir_model, args.resume + ".pth")
        print("==> Load checkpoint: {}".format(backup_pth))
        ps_net.load_state_dict(torch.load(backup_pth), strict=False)
    else:
        print('==> Train from scratch')

    ps_net = ps_net.to(f"cuda:{args.device}")

    optimizer = torch.optim.Adam(ps_net.parameters(), args.lr)

    train(ps_net, optimizer, train_dataloader, val_dataloader, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--idx', type=int, default=1, help='Index to identify models.')
    parser.add_argument('--dataset', type=str, default="CAVE", help='Dataset to be loaded.')
    parser.add_argument('--train_size', type=int, default=160,
                        help='Size of a MS image in a batch, usually 4x less than pan.')
    parser.add_argument('--msfa_size', type=int, default=4, help='Size of MSFA')
    parser.add_argument('--spatial_ratio', type=int, default=2, help='Ratio of spatial resolutions between MS and PAN')
    parser.add_argument('--num_bands', type=int, default=16, help='Number of bands of a MS image.')
    parser.add_argument('--stride', type=int, default=32, help='Stride when crop an original image into patches.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training dataset.')
    parser.add_argument('--epochs', type=int, default=1200, help='Total epochs to train the model.')
    parser.add_argument('--iters_per_epoch', type=int, default=100, help='Iteration steps per epoch.')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='Save the checkpoints of the model every [save_freq] epochs.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate to train the model.')
    parser.add_argument('--lr_decay', action="store_true", help='Determine if to decay the learning rate.')
    parser.add_argument('--device', type=str, default='0', help='Device to train the model.')
    parser.add_argument('--num_workers', type=int, default=1, help='Num_workers to train the model.')
    parser.add_argument('--resume', type=str, default='', help='Index of the model to be resumed, eg. 1000.')
    parser.add_argument('--resume_demosaic', type=str, default='', help='Index of the demosaicing model to be resumed, eg. 1000.')
    parser.add_argument('--data_path', type=str, default="../DataSet/", help='Path of the dataset.')
    parser.add_argument('--record', type=bool, default=True, help='Whether to record the PSNR of each epoch.')
    parser.add_argument('--visual', action='store_true', default=False, help='Whether to visualize the validation data.')
    parser.add_argument('--visual_freq', type=int, default=50, help='Frequency to visualize the validation data.')

    args = parser.parse_args()
    main(args)