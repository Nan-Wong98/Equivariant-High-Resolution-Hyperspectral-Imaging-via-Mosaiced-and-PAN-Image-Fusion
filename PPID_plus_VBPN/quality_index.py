import numpy
import torch
import utils

def calc_psnr(hrms, fused):
    mse = torch.mean((hrms - fused) ** 2, [0,2,3])
    psnr = 10*torch.mean(torch.log10(hrms.max(0)[0].max(-1)[0].max(-1)[0] / mse))
    return psnr

def gaussian(window_size, sigma):  
    gauss = torch.Tensor([torch.exp(-torch.Tensor([(x - window_size//2)**2/float(2*sigma**2)])) for x in range(window_size)])
    return gauss/gauss.sum()  
  
def create_window(window_size, channel, sigma):  
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)  
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)  
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()  
    return window  
  
def calc_ssim(img1, img2, window_size=11, sigma=1.5, size_average=True):  
    (_, channel, _, _) = img1.size()  
    window = create_window(window_size, channel, sigma)  
      
    mu1 = torch.nn.functional.conv2d(img1, window, padding=window_size//2, groups=channel)  
    mu2 = torch.nn.functional.conv2d(img2, window, padding=window_size//2, groups=channel)  
      
    mu1_sq = mu1.pow(2)  
    mu2_sq = mu2.pow(2)  
    mu1_mu2 = mu1 * mu2  
      
    sigma1_sq = torch.nn.functional.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq  
    sigma2_sq = torch.nn.functional.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq  
    sigma12 = torch.nn.functional.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2  
      
    C1 = 0.01**2  
    C2 = 0.03**2  
      
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))  
      
    if size_average:  
        return ssim_map.mean()  
    else:  
        return ssim_map.mean(1).mean(1).mean(1) 
  
def calc_sam(hrms, fused):
    dot_product = torch.sum(hrms * fused, dim=1)
      
    magnitude1 = torch.norm(hrms, dim=1)
    magnitude2 = torch.norm(fused, dim=1)  
       
    epsilon = 1e-10
    magnitude1 = torch.clamp(magnitude1, min=epsilon)  
    magnitude2 = torch.clamp(magnitude2, min=epsilon)  
       
    cosine_angle = dot_product / (magnitude1 * magnitude2)  

    angle = torch.acos(cosine_angle).mean() * 180 / torch.pi
      
    return angle

def calc_ergas(hrms, fused, scale_factor=4):
    mse = torch.mean((hrms - fused) ** 2, [0,2,3])
    mean = torch.mean(hrms, [0, 2, 3])**2
    ergas = 100/scale_factor*torch.sqrt(torch.mean(mse / mean))
    
    return ergas

def calc_qi(img1, img2, patch_size=16, average=True):  
    # crop the image
    img1 = img1[:, :, :img1.shape[2]//patch_size*patch_size, :img1.shape[3]//patch_size*patch_size]
    img2 = img2[:, :, :img2.shape[2]//patch_size*patch_size, :img2.shape[3]//patch_size*patch_size]

    # Get the dimensions of the tensors  
    N, C, H, W = img1.shape
      
    # Reshape tensors to add patch dimensions  
    img1 = img1.view(N, C, H // patch_size, patch_size, W // patch_size, patch_size)  
    img2 = img2.view(N, C, H // patch_size, patch_size, W // patch_size, patch_size)  
      
    # Compute mean, variance, and covariance for each patch  
    mean1 = img1.mean(dim=(3, 5), keepdim=True)  # Mean over height and width within each patch  
    mean2 = img2.mean(dim=(3, 5), keepdim=True)  
    var1 = torch.var(img1, dim=(3, 5))  # Variance over height and width within each patch  
    var2 = torch.var(img2, dim=(3, 5))  
    covariance = ((img1 - mean1) * (img2 - mean2)).mean(dim=(3, 5))  # Covariance over height and width within each patch  
    
    mean1 = mean1.squeeze(3, 5)
    mean2 = mean2.squeeze(3, 5)
    # Compute the quality index
    numerator = 4 * covariance * mean1 * mean2
    denominator = (var1 + var2) * (mean1 ** 2 + mean2 ** 2) + 1e-4
    quality_index = numerator / denominator
      
    # Remove the extra dimensions added for patches  
    if average == True:
        quality_index = quality_index.mean() 
    else:
        quality_index = quality_index.mean((0,2,3))
      
    return quality_index

def calc_d_lambda(hrms, mosaic, patch_size=4, p=1.):
    c = mosaic.shape[1]

    d_lambda = 0
    for i in range(c):
        mosaic_bandi = mosaic[:, i].repeat(1, c-1, 1, 1)
        mosaic_bands = torch.cat((mosaic[:, :i], mosaic[:, i+1:]), 1)
        hrms_bandi = hrms[:, i].repeat(1, c-1, 1, 1)
        hrms_bands = torch.cat((hrms[:, :i], hrms[:, i+1:]), 1)
        d_lambda_tmp = calc_qi(mosaic_bandi, mosaic_bands, patch_size=patch_size, average=False)\
                        - calc_qi(hrms_bandi, hrms_bands, patch_size=patch_size, average=False)
        d_lambda += torch.mean(torch.pow(torch.abs(d_lambda_tmp), p))
    d_lambda /= c
    d_lambda = torch.pow(d_lambda, 1./p)

    return d_lambda

def get_gaussian_kernel(kernel_size=5, sigma=1.0):
    """Create a 2D Gaussian kernel."""
    x = torch.arange(-kernel_size//2 + 1, kernel_size//2 + 1, dtype=torch.float64)
    gauss = torch.exp(-(x**2) / (2 * sigma**2))
    kernel = gauss.ger(gauss)  # Outer product
    kernel /= kernel.sum()      # Normalize
    return kernel.view(1, 1, kernel_size, kernel_size)  # (1, 1, H, W)

def calc_d_s(hrms, pan, patch_size=4, scale_factor=2, q=1.):
    # qi
    pan_extend = pan.repeat(1, hrms.shape[1], 1, 1)

    hrms_down = torch.nn.functional.interpolate(hrms, scale_factor=1/scale_factor, mode='bicubic')
    pan_down = torch.nn.functional.interpolate(pan, scale_factor=1/scale_factor, mode='bicubic')
    pan_down = pan_down.repeat(1, hrms.shape[1], 1, 1)

    qi = calc_qi(hrms, pan_extend, patch_size=patch_size, average=False) - calc_qi(hrms_down, pan_down, patch_size=patch_size, average=False)

    d_s = torch.pow(torch.mean(torch.pow(torch.abs(qi), q)), 1./q)

    return d_s

def calc_qnr_mosaic(hrms, mosaic, pan, msfa_kernel, patch_size=32, scale_factor=2, p=1, q=1., alpha=1, beta=3):
    gaussian_kernel = get_gaussian_kernel(5, 1).to(hrms.device)
    hrms = torch.nn.functional.conv2d(hrms, gaussian_kernel.repeat(hrms.shape[1], 1, 1, 1), stride=1, padding=gaussian_kernel.shape[-1]//2, groups=hrms.shape[1])
    mosaic = torch.nn.functional.pixel_unshuffle(mosaic, downscale_factor=msfa_kernel.shape[2]//2)
    mosaic = torch.nn.functional.conv2d(mosaic, gaussian_kernel.repeat(hrms.shape[1], 1, 1, 1), stride=1, padding=gaussian_kernel.shape[-1]//2, groups=hrms.shape[1])
    pan = torch.nn.functional.conv2d(pan, gaussian_kernel, stride=1, padding=gaussian_kernel.shape[-1]//2)
    
    d_lambda = calc_d_lambda(hrms, mosaic, patch_size=patch_size, p=p)
    d_s = calc_d_s(hrms, pan, patch_size=patch_size, scale_factor=scale_factor, q=q)

    qnr_mosaic = torch.pow(1-d_lambda, alpha) * torch.pow(1-d_s, beta)

    return qnr_mosaic, d_lambda, d_s

if __name__ == '__main__':
    import copy
    a_torch = torch.rand(1, 16, 256, 256)
    b_torch = a_torch + 0.1*torch.rand(1, 16, 256, 256)

    psnr = calc_psnr(a_torch, b_torch)
    print(psnr)

    ssim = calc_ssim(a_torch, b_torch)
    print(ssim)

    sam = calc_sam(a_torch, b_torch)
    print(sam)

    ergas = calc_ergas(a_torch, b_torch)
    print(ergas)

    MSFA = numpy.array([[0, 1, 2, 3],
                        [4, 5, 6, 7],
                        [8, 9, 10, 11],
                        [12, 13, 14, 15]])
    msfa_kernel = torch.zeros(MSFA.shape[0] * MSFA.shape[1], 1, MSFA.shape[0]*2, MSFA.shape[1]*2)
    for i in range(MSFA.shape[0]):
        for j in range(MSFA.shape[1]):
            msfa_kernel[int(MSFA[i, j]), 0, i, j] += 1

    c_torch = torch.rand(1, 1, 128, 128)

    d_torch = torch.rand(1, 1, 256, 256)

    qnr_mosaic, d_lambda, d_s = calc_qnr_mosaic(copy.deepcopy(a_torch), copy.deepcopy(c_torch), copy.deepcopy(d_torch), msfa_kernel, patch_size=32, scale_factor=2, p=1., q=1., alpha=1., beta=1.)

    print(qnr_mosaic.item(), d_lambda.item(), d_s.item())