import numpy
import torch
import utils
import torch.nn.functional as F

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

def calc_q2n(I_GT, I_F, Q_blocks_size, Q_shift):
    """
    Q2n index calculation for image fusion quality assessment.
    
    Args:
        I_GT: Ground-Truth image (torch.Tensor), shape [C, H, W]
        I_F: Fused Image (torch.Tensor), shape [C, H, W]
        Q_blocks_size: Block size for local Q-index calculation
        Q_shift: Block shift for local Q-index calculation
        
    Returns:
        Q2n_index: Overall Q2n index value
        Q2n_index_map: Map of Q2n values, shape [stepx, stepy]
    """
    # Ensure input tensors are float and in CHW format
    I_GT = I_GT.float()
    I_F = I_F.float()
    
    C, N1, N2 = I_GT.shape
    size2 = Q_blocks_size
    
    stepx = (N1 + Q_shift - 1) // Q_shift  # ceil division
    stepy = (N2 + Q_shift - 1) // Q_shift
    
    if stepy <= 0:
        stepy = 1
        stepx = 1
    
    est1 = (stepx - 1) * Q_shift + Q_blocks_size - N1
    est2 = (stepy - 1) * Q_shift + Q_blocks_size - N2
    
    # Pad images if needed
    if est1 > 0 or est2 > 0:
        # Pad with reflection
        pad_width = (est2 // 2, est2 - est2 // 2, est1 // 2, est1 - est1 // 2)
        I_GT = F.pad(I_GT, pad_width, mode='reflect')
        I_F = F.pad(I_F, pad_width, mode='reflect')
        
    C, N1, N2 = I_GT.shape
    
    # Ensure number of channels is power of 2
    if (torch.log2(torch.tensor(C)) % 1) != 0:
        Ndif = 2**torch.ceil(torch.log2(torch.tensor(C))).int() - C
        if Ndif > 0:
            pad_channels = torch.zeros(Ndif, N1, N2, device=I_GT.device, dtype=I_GT.dtype)
            I_GT = torch.cat([I_GT, pad_channels], dim=0)
            I_F = torch.cat([I_F, pad_channels], dim=0)
    
    C, N1, N2 = I_GT.shape
    
    # Initialize Q2n map
    valori = torch.zeros(stepx, stepy, C, device=I_GT.device, dtype=torch.float32)
    
    # Process each block
    for j in range(stepx):
        for i in range(stepy):
            # Calculate block coordinates
            x_start = j * Q_shift
            x_end = x_start + Q_blocks_size
            y_start = i * Q_shift
            y_end = y_start + size2
            
            # Extract blocks
            block_GT = I_GT[:, x_start:x_end, y_start:y_end]
            block_F = I_F[:, x_start:x_end, y_start:y_end]
            
            # Calculate onion quality for this block
            o = onions_quality(block_GT, block_F, Q_blocks_size)
            valori[j, i, :] = o
    
    # Calculate Q2n map and index
    Q2n_index_map = torch.sqrt(torch.sum(valori**2, dim=2))
    Q2n_index = torch.mean(Q2n_index_map)
    
    return Q2n_index, Q2n_index_map


def onions_quality(dat1, dat2, size1):
    """
    Auxiliary function for onions quality calculation.
    
    Args:
        dat1: Ground truth block (C, H, W)
        dat2: Fused image block (C, H, W)
        size1: Block size
        
    Returns:
        q: Quality score for the block
    """
    # Modify dat2: first channel remains, others are negated
    dat2_mod = dat2.clone()
    if dat2_mod.shape[0] > 1:
        dat2_mod[1:, :, :] = -dat2_mod[1:, :, :]
    
    C, H, W = dat1.shape
    
    # Block normalization
    dat1_norm = []
    dat2_norm = []
    
    for i in range(C):
        a1 = dat1[i:i+1, :, :]
        a1_norm, s, t = norm_blocco(a1)
        dat1_norm.append(a1_norm)
        
        # Normalize dat2
        a2 = dat2_mod[i:i+1, :, :]
        if s == 0:
            if i == 0:
                a2_norm = a2 - s + 1
            else:
                a2_norm = -(-a2 - s + 1)
        else:
            if i == 0:
                a2_norm = ((a2 - s) / t) + 1
            else:
                a2_norm = -(((-a2 - s) / t) + 1)
        
        dat2_norm.append(a2_norm)
    
    dat1 = torch.cat(dat1_norm, dim=0)
    dat2 = torch.cat(dat2_norm, dim=0)
    
    # Calculate means
    m1 = torch.mean(dat1, dim=(1, 2))
    m2 = torch.mean(dat2, dim=(1, 2))
    
    # Calculate moduli
    mod_q1m = torch.sqrt(torch.sum(m1**2))
    mod_q2m = torch.sqrt(torch.sum(m2**2))
    
    mod_q1 = torch.sqrt(torch.sum(dat1**2, dim=0))
    mod_q2 = torch.sqrt(torch.sum(dat2**2, dim=0))
    
    # Calculate terms
    termine2 = mod_q1m * mod_q2m
    termine4 = mod_q1m**2 + mod_q2m**2
    
    int1 = (H*W) / ((H*W) - 1) * torch.mean(mod_q1**2)
    int2 = (H*W) / ((H*W) - 1) * torch.mean(mod_q2**2)
    termine3 = int1 + int2 - (H*W) / ((H*W) - 1) * (mod_q1m**2 + mod_q2m**2)
    
    mean_bias = 2 * termine2 / termine4
    
    if torch.isclose(termine3, torch.tensor(0.0)):
        q = torch.zeros(C, device=dat1.device)
        q[-1] = mean_bias
    else:
        cbm = 2 / termine3
        
        # Onion multiplication
        qu = onion_mult2D(dat1, dat2)
        qm = onion_mult(m1, m2)
        
        qv = torch.zeros(C, device=dat1.device)
        for i in range(C):
            qv[i] = (H*W) / ((H*W) - 1) * torch.mean(qu[i, :, :])
        
        q = qv - (H*W) / ((H*W) - 1) * qm
        q = q * mean_bias * cbm
    
    return q


def norm_blocco(block):
    """
    Normalize a single channel block.
    
    Args:
        block: Single channel block (1, H, W)
        
    Returns:
        norm_block: Normalized block
        s: Mean value
        t: Standard deviation
    """
    s = torch.mean(block)
    t = torch.std(block, unbiased=False)
    
    if t == 0:
        norm_block = block - s + 1
    else:
        norm_block = (block - s) / t + 1
    
    return norm_block, s, t


def onion_mult2D(onion1, onion2):
    """
    2D onion multiplication for multi-channel tensors.
    
    Args:
        onion1: First tensor (C, H, W)
        onion2: Second tensor (C, H, W)
        
    Returns:
        ris: Result of onion multiplication
    """
    C, H, W = onion1.shape
    
    if C == 1:
        return onion1 * onion2
    
    L = C // 2
    a = onion1[:L, :, :]
    b = onion1[L:, :, :]
    b = torch.cat([b[:1, :, :], -b[1:, :, :]], dim=0) if L > 1 else b
    
    c = onion2[:L, :, :]
    d = onion2[L:, :, :]
    d = torch.cat([d[:1, :, :], -d[1:, :, :]], dim=0) if L > 1 else d
    
    if C == 2:
        ris = torch.cat([a*c - d*b, a*d + c*b], dim=0)
    else:
        ris1 = onion_mult2D(a, c)
        ris2 = onion_mult2D(d, torch.cat([b[:1, :, :], -b[1:, :, :]], dim=0) if L > 1 else b)
        ris3 = onion_mult2D(torch.cat([a[:1, :, :], -a[1:, :, :]], dim=0) if L > 1 else a, d)
        ris4 = onion_mult2D(c, b)
        
        aux1 = ris1 - ris2
        aux2 = ris3 + ris4
        ris = torch.cat([aux1, aux2], dim=0)
    
    return ris


def onion_mult(onion1, onion2):
    """
    Onion multiplication for vectors.
    
    Args:
        onion1: First vector (C,)
        onion2: Second vector (C,)
        
    Returns:
        ris: Result of onion multiplication
    """
    N = len(onion1)
    
    if N == 1:
        return onion1 * onion2
    
    L = N // 2
    a = onion1[:L]
    b = onion1[L:]
    b = torch.cat([b[:1], -b[1:]]) if L > 1 else b
    
    c = onion2[:L]
    d = onion2[L:]
    d = torch.cat([d[:1], -d[1:]]) if L > 1 else d
    
    if N == 2:
        ris = torch.cat([a*c - d*b, a*d + c*b])
    else:
        ris1 = onion_mult(a, c)
        ris2 = onion_mult(d, torch.cat([b[:1], -b[1:]]) if L > 1 else b)
        ris3 = onion_mult(torch.cat([a[:1], -a[1:]]) if L > 1 else a, d)
        ris4 = onion_mult(c, b)
        
        aux1 = ris1 - ris2
        aux2 = ris3 + ris4
        ris = torch.cat([aux1, aux2])
    
    return ris

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