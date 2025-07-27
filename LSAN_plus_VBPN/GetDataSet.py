import numpy
import os
import scipy.io as scio
import cv2
import torch
from torch.utils.data import Dataset
import pickle
import tqdm
from tqdm.contrib import tzip
import h5py
import utils
from scipy import signal
import random
import math
import scipy

def crop_to_patch(img, size, stride):
    H, W = img.shape[:2]
    patches = []
    for h in range(0, H, stride):
        for w in range(0, W, stride):
            if h + size <= H and w + size <= W:
                patch = img[h: h + size, w: w + size, :]
                patches.append(patch)
    return patches

def shifted_anisotropic_Gaussian(k_size=21, sf=4, lambda_1=1.2, lambda_2=5., theta=0, shift=True):
    '''
    # modified version of https://github.com/cszn/USRNet/blob/master/utils/utils_sisr.py
    '''
    # set covariance matrix
    Lam = numpy.diag([lambda_1, lambda_2])
    U = numpy.array([[numpy.cos(theta), -numpy.sin(theta)],
                    [numpy.sin(theta),  numpy.cos(theta)]])
    sigma = U @ Lam @ U.T                                 # 2 x 2
    inv_sigma = numpy.linalg.inv(sigma)[None, None, :, :]    # 1 x 1 x 2 x 2

    # set expectation position (shifting kernel for aligned image)
    if shift:
        center = k_size // 2 + 0.5*(sf - k_size % 2)
    else:
        center = k_size // 2

    # Create meshgrid for Gaussian
    X, Y = numpy.meshgrid(range(k_size), range(k_size))
    Z = numpy.stack([X, Y], 2).astype(numpy.float32)[:, :, :, None]                  # k x k x 2 x 1

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z - center
    ZZ_t = ZZ.transpose(0,1,3,2)
    ZZZ = -0.5 * numpy.squeeze(ZZ_t @ inv_sigma  @ ZZ).reshape([1, -1])
    kernel = scipy.special.softmax(ZZZ, axis=1).reshape([k_size, k_size]) # k x k

    # The convariance of the marginal distributions along x and y axis
    s1, s2 = sigma[0, 0], sigma[1, 1]
    # Pearson corrleation coefficient
    rho = sigma[0, 1] / (math.sqrt(s1) * math.sqrt(s2))
    kernel_infos = numpy.array([s1, s2, rho])   # (3,)

    return kernel, kernel_infos

def sigma_estimate(im_noisy, im_gt, win, sigma_spatial):
    noise2 = (im_noisy - im_gt) ** 2
    sigma2_map_est = cv2.GaussianBlur(noise2, (win, win), sigma_spatial)
    sigma2_map_est = sigma2_map_est.astype(numpy.float32)
    sigma2_map_est = numpy.where(sigma2_map_est<1e-10, 1e-10, sigma2_map_est)
    if sigma2_map_est.ndim == 2:
        sigma2_map_est = sigma2_map_est[:, :, numpy.newaxis]
    return sigma2_map_est

class MakeSimulateDatasetforDemosaic(Dataset):
    def __init__(self, args, type="train"):
        cache_path = os.path.join(args.cache_path, type + "_cache.pkl")
        self.train_size = args.train_size
        if not os.path.exists(cache_path):
            self.mosaic, self.target = [], []
            base_path = os.path.join(args.data_path, args.dataset, type)
            print("Cache file not found. Generate it from: ", base_path)
            hrms_imgs = os.listdir(base_path)

            for hrms_name in tqdm.tqdm(hrms_imgs):
                if args.dataset == "CAVE":
                    hrms = scio.loadmat(os.path.join(base_path, hrms_name))["b"]
                elif args.dataset == "ICVL":
                    hrms = h5py.File(os.path.join(base_path, hrms_name))["rad"][:]
                    hrms = numpy.rot90(hrms.transpose(2, 1, 0))
                    hrms /= hrms.max((0, 1))
                hrms_select_bands = hrms[:hrms.shape[0]//(args.msfa_size*args.spatial_ratio)*(args.msfa_size*args.spatial_ratio),
                                        :hrms.shape[1]//(args.msfa_size*args.spatial_ratio)*(args.msfa_size*args.spatial_ratio),
                                        12:28].astype(numpy.float32)

                MSFA = numpy.array([[0, 1, 2, 3],
                                    [4, 5, 6, 7],
                                    [8, 9, 10, 11],
                                    [12, 13, 14, 15]])
                # MS simulate
                # downsampling
                ms_blur_tensor = torch.from_numpy(hrms_select_bands).permute(2, 0, 1).unsqueeze(0)
                lrms_tensor = torch.nn.functional.avg_pool2d(ms_blur_tensor, 2, 2)
                lrms = lrms_tensor[0].permute(1, 2, 0).numpy()

                # mosaicing
                mosaic = utils.MSFA_filter(lrms, MSFA)

                if type == "train":
                    mosaic_patches = crop_to_patch(mosaic, args.train_size, args.stride)

                    self.mosaic += mosaic_patches
                elif type == "test":
                    self.mosaic.append(mosaic)
                    self.target.append(lrms)

            with open(cache_path, "wb") as f:
                pickle.dump([self.mosaic, self.target], f)

        print("Load data from cache file: ", cache_path)
        with open(cache_path, "rb") as f:
            self.mosaic, self.target = pickle.load(f)

    def __len__(self):
        return len(self.mosaic)

    def __getitem__(self, index):
        if self.target == []:
            mosaic = torch.from_numpy(self.mosaic[index].astype(numpy.float32))
            mosaic = mosaic.permute(2, 0, 1)

            return mosaic
        else:
            mosaic = torch.from_numpy(self.mosaic[index].astype(numpy.float32))
            mosaic = mosaic.permute(2, 0, 1)
            target = torch.from_numpy(self.target[index].astype(numpy.float32))
            target = target.permute(2, 0, 1)

            return mosaic, target
        
class MakeRealWorldDatasetforDemosaic(Dataset):
    def __init__(self, args, type="train"):
        cache_path = os.path.join(args.cache_path, type + "_cache.pkl")
        self.train_size = args.train_size
        if not os.path.exists(cache_path):
            self.mosaic = []
            base_path = os.path.join(args.data_path, args.dataset, type)
            print("Cache file not found. Generate it from: ", base_path)
            mosaic_path = os.path.join(base_path, "mosaic")
            pan_path = os.path.join(base_path, "pan")
            mosaic_imgs = os.listdir(mosaic_path)
            mosaic_imgs.sort()
            pan_imgs = os.listdir(pan_path)
            pan_imgs.sort()
            assert len(mosaic_imgs) == len(pan_imgs), "Length mismatch between MS and PAN images!"

            for mosaic_name, pan_name in tqdm.tqdm(tzip(mosaic_imgs, pan_imgs)):
                with open(os.path.join(mosaic_path, mosaic_name), "rb") as f:
                    mosaic_raw = numpy.frombuffer(f.read(), dtype=numpy.int16)
                    mosaic = numpy.zeros((255*4, 276*4, 1))
                    for i in range(args.msfa_size):
                        for j in range(args.msfa_size):
                            mosaic[i::args.msfa_size, j::args.msfa_size, 0] = mosaic_raw[255*276*(i*args.msfa_size+j): 255*276*(i*args.msfa_size+j+1)].reshape(255, 276)
                
                mosaic = mosaic / 2**12
                
                if type == "train":
                    mosaic_patches = crop_to_patch(mosaic, args.train_size, args.stride)

                    self.mosaic += mosaic_patches
                elif type == "test":
                    self.mosaic.append(mosaic)

            with open(cache_path, "wb") as f:
                pickle.dump(self.mosaic, f)

        print("Load data from cache file: ", cache_path)
        with open(cache_path, "rb") as f:
            self.mosaic = pickle.load(f)

    def __len__(self):
        return len(self.mosaic)

    def __getitem__(self, index):
        mosaic = torch.from_numpy(self.mosaic[index].astype(numpy.float32)).permute(2, 0, 1)
        return mosaic

def input_matrix_wpn(inH, inW, msfa_size):

    h_offset_coord = torch.zeros(inH, inW, 1)
    w_offset_coord = torch.zeros(inH, inW, 1)
    for i in range(0,msfa_size):
        h_offset_coord[i::msfa_size, :, 0] = (i+1)/msfa_size
        w_offset_coord[:, i::msfa_size, 0] = (i+1)/msfa_size

    pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)
    pos_mat = pos_mat.contiguous().view(1, -1, 2)
    return pos_mat

class MakeSimulateDatasetforPansharpening(Dataset):
    def __init__(self, args, type="train", demosaic_net=None):
        self.train_size = args.train_size
        cache_path = os.path.join(args.cache_path, type + "_cache.pkl")
        if not os.path.exists(cache_path):
            self.lrms, self.pan, self.hrms = [], [], []
            mosaic_noise_list, pan_noise_list = [], []
            base_path = os.path.join(args.data_path, args.dataset, type)
            print("Cache file not found. Generate it from: ", base_path)
            hrms_imgs = os.listdir(base_path)

            for hrms_name in tqdm.tqdm(hrms_imgs):
                if args.dataset == "CAVE":
                    hrms = scio.loadmat(os.path.join(base_path, hrms_name))["b"]
                elif args.dataset == "ICVL":
                    hrms = h5py.File(os.path.join(base_path, hrms_name))["rad"][:]
                    hrms = numpy.rot90(hrms.transpose(2, 1, 0))
                    hrms /= hrms.max((0, 1))
                hrms_select_bands = hrms[:hrms.shape[0]//(args.msfa_size*args.spatial_ratio)*(args.msfa_size*args.spatial_ratio),
                                        :hrms.shape[1]//(args.msfa_size*args.spatial_ratio)*(args.msfa_size*args.spatial_ratio),
                                        12:28].astype(numpy.float32)

                MSFA = numpy.array([[0, 1, 2, 3],
                                    [4, 5, 6, 7],
                                    [8, 9, 10, 11],
                                    [12, 13, 14, 15]])
                # MS simulate
                # downsampling
                ms_blur_tensor = torch.from_numpy(hrms_select_bands).permute(2, 0, 1).unsqueeze(0)
                lrms_tensor = torch.nn.functional.avg_pool2d(ms_blur_tensor, 2, 2)
                lrms = lrms_tensor[0].permute(1, 2, 0).numpy()

                # mosaicing
                mosaic = utils.MSFA_filter(lrms, MSFA)

                mosaic_tensor = torch.from_numpy(mosaic.astype(numpy.float32)).permute(2, 0, 1).unsqueeze(0).to(f"cuda:{args.device}")
                with torch.no_grad():
                    scale_coord_map = input_matrix_wpn(mosaic_tensor.shape[2], mosaic_tensor.shape[3], MSFA.shape[0]).to(mosaic_tensor.device)
                    mosaic_up = torch.zeros(mosaic_tensor.shape[0], MSFA.shape[0]*MSFA.shape[1], mosaic_tensor.shape[2], mosaic_tensor.shape[3]).to(mosaic_tensor.device)
                    for i in range(MSFA.shape[0]):
                        for j in range(MSFA.shape[1]):
                            mosaic_up[:, i*MSFA.shape[1]+j, i::MSFA.shape[0], j::MSFA.shape[1]] = mosaic_tensor[:, 0, i::MSFA.shape[0], j::MSFA.shape[1]]
                    demosaic = demosaic_net([mosaic_up, mosaic_tensor], scale_coord_map).detach()
                demosaic = demosaic[0].cpu().permute(1, 2, 0).numpy()
                
                ## PAN simulate
                spe_res = numpy.array([1., 1, 2, 4, 8, 9, 10, 12, 16, 12, 10, 9, 7, 3, 2, 1])
                spe_res /= spe_res.sum()
                pan = numpy.sum(hrms_select_bands * spe_res, axis=-1, keepdims=True)

                spatial_ratio = pan.shape[0] // demosaic.shape[0]
                if type == "train":
                    demosaic_patches = crop_to_patch(demosaic, args.train_size//spatial_ratio, args.stride//spatial_ratio)
                    pan_patches = crop_to_patch(pan, args.train_size, args.stride)
                    self.lrms += demosaic_patches
                    self.pan += pan_patches
                elif type == "test":
                    self.lrms.append(demosaic)
                    self.pan.append(pan)
                    self.hrms.append(hrms_select_bands)
            
            with open(cache_path, "wb") as f:
                pickle.dump([self.lrms, self.pan, self.hrms], f)

        print("Load data from cache file: ", cache_path)
        with open(cache_path, "rb") as f:
            self.lrms, self.pan, self.hrms = pickle.load(f)

    def __len__(self):
        return len(self.lrms)

    def __getitem__(self, index):
        if self.hrms != []:
            lrms = torch.from_numpy(self.lrms[index].astype(numpy.float32))
            lrms = lrms.permute(2, 0, 1)
            pan = torch.from_numpy(self.pan[index].astype(numpy.float32))
            pan = pan.permute(2, 0, 1)
            hrms = torch.from_numpy(self.hrms[index].astype(numpy.float32))
            hrms = hrms.permute(2, 0, 1)
            return lrms, pan, hrms
        else:
            lrms, pan = self.lrms[index], self.pan[index]
            spatial_ratio = pan.shape[0] // lrms.shape[0]
            # for MS
            # blur kernel 
            lam1 = random.uniform(0.2, spatial_ratio)
            lam2 = random.uniform(lam1, spatial_ratio) if random.random() < 0.7 else lam1
            theta = random.uniform(0, numpy.pi)
            kernel, kernel_infos = shifted_anisotropic_Gaussian(k_size=7,
                                                                sf=2,
                                                                lambda_1=lam1**2,
                                                                lambda_2=lam2**2,
                                                                theta=theta,
                                                                shift=False)
            
            # blurring
            ms_blur = scipy.ndimage.convolve(lrms, kernel[:, :, numpy.newaxis], mode='reflect')
            ms_blur = numpy.clip(ms_blur, a_min=0.0, a_max=1.0)
            
            # downsampling
            ms_blur = cv2.resize(ms_blur, dsize=None, fx=1/spatial_ratio, fy=1/spatial_ratio, interpolation=cv2.INTER_CUBIC)

            # adding noise
            std = random.uniform(0.1, 15) / 255.0
            ms_lr = ms_blur + torch.randn(ms_blur.shape, dtype=torch.float32).numpy() * std
            ms_lr = numpy.clip(ms_lr, a_min=0, a_max=1.0)

            ms_label = torch.from_numpy(lrms.transpose([2,0,1])).type(torch.float32)        # c x h x w
            ms_lr = torch.from_numpy(ms_lr.transpose([2,0,1])).type(torch.float32)          # c x h x w
            ms_blur = torch.from_numpy(ms_blur.transpose([2,0,1])).type(torch.float32)      # c x h x w
            kernel_infos = torch.from_numpy(kernel_infos).type(torch.float32)               # 3
            nlevel = torch.tensor([std], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)   # 1 x 1 x 1

            # for PAN
            # downsampling
            pan_label_down = cv2.resize(pan, dsize=None, fx=1/spatial_ratio, fy=1/spatial_ratio, interpolation=cv2.INTER_CUBIC)[:, :, numpy.newaxis]
            pan_n = pan_label_down

            std = random.uniform(0.1, 30) / 255.0
            pan_n = pan_n + torch.randn(pan_n.shape, dtype=torch.float32).numpy() * std
            pan_n = numpy.clip(pan_n, a_min=0, a_max=1.0)

            pan_eps2 = 1e-5
            pan_sigma_spatial = 3
            pan_window_size = 2 * pan_sigma_spatial + 1
            sigma2_map_est = sigma_estimate(im_noisy=pan_n, im_gt=pan_label_down, win=pan_window_size,
                                            sigma_spatial=pan_sigma_spatial)
            tensor_pan_label = torch.from_numpy(pan_label_down.transpose([2,0,1])).type(torch.float32)
            tensor_pan_n = torch.from_numpy(pan_n.transpose([2,0,1])).type(torch.float32)
            tensor_map = torch.from_numpy(sigma2_map_est.transpose([2,0,1])).type(torch.float32)
            tensor_eps2 = torch.tensor([pan_eps2], dtype=torch.float32).reshape((1, 1, ))
            
            return ms_label, ms_lr, ms_blur, kernel_infos, nlevel, \
            tensor_pan_label, tensor_pan_n, tensor_map, tensor_eps2

class MakeRealWorldDatasetForPansharpening(Dataset):
    def __init__(self, args, type="train", demosaic_net=None):
        self.train_size = args.train_size
        cache_path = os.path.join(args.cache_path, type + "_cache.pkl")
        if not os.path.exists(cache_path):
            self.mosaic, self.lrms, self.pan = [], [], []
            base_path = os.path.join(args.data_path, args.dataset, type)
            print("Cache file not found. Generate it from: ", base_path)
            mosaic_path = os.path.join(base_path, "mosaic")
            pan_path = os.path.join(base_path, "pan")
            mosaic_imgs = os.listdir(mosaic_path)
            mosaic_imgs.sort()
            pan_imgs = os.listdir(pan_path)
            pan_imgs.sort()
            assert len(mosaic_imgs) == len(pan_imgs), "Length mismatch between MS and PAN images!"

            for mosaic_name, pan_name in tqdm.tqdm(tzip(mosaic_imgs, pan_imgs)):
                with open(os.path.join(mosaic_path, mosaic_name), "rb") as f:
                    mosaic_raw = numpy.frombuffer(f.read(), dtype=numpy.int16)
                    mosaic = numpy.zeros((255*4, 276*4, 1))
                    for i in range(args.msfa_size):
                        for j in range(args.msfa_size):
                            mosaic[i::args.msfa_size, j::args.msfa_size, 0] = mosaic_raw[255*276*(i*args.msfa_size+j): 255*276*(i*args.msfa_size+j+1)].reshape(255, 276)
                with open(os.path.join(pan_path, pan_name), "rb") as f:
                    pan_raw = numpy.frombuffer(f.read(), dtype=numpy.int16)
                    pan = pan_raw.reshape((255*8, 276*8, 1))
                assert pan.shape[1] % mosaic.shape[1] == 0 and pan.shape[2] % mosaic.shape[
                    2] == 0, "Mismatch between the spatial size of MS and PAN"

                mosaic = mosaic / 2**12
                pan = pan / 2**12

                mosaic_tensor = torch.from_numpy(mosaic.astype(numpy.float32)).permute(2, 0, 1).unsqueeze(0).to(f"cuda:{args.device}")
                with torch.no_grad():
                    scale_coord_map = input_matrix_wpn(mosaic_tensor.shape[2], mosaic_tensor.shape[3], 4).to(mosaic_tensor.device)
                    mosaic_up = torch.zeros(mosaic_tensor.shape[0], 4*4, mosaic_tensor.shape[2], mosaic_tensor.shape[3]).to(mosaic_tensor.device)
                    for i in range(4):
                        for j in range(4):
                            mosaic_up[:, i*4+j, i::4, j::4] = mosaic_tensor[:, 0, i::4, j::4]
                    demosaic = demosaic_net([mosaic_up, mosaic_tensor], scale_coord_map).detach()[0].cpu().numpy().transpose(1, 2, 0)

                spatial_ratio = pan.shape[0] // demosaic.shape[0]
                if type == "train":
                    demosaic_patches = crop_to_patch(demosaic, args.train_size//spatial_ratio, args.stride//spatial_ratio)
                    pan_patches = crop_to_patch(pan, args.train_size, args.stride)

                    self.lrms += demosaic_patches
                    self.pan += pan_patches

                elif type == "test":
                    self.mosaic.append(mosaic)
                    self.lrms.append(demosaic)
                    self.pan.append(pan)

            with open(cache_path, "wb") as f:
                pickle.dump([self.mosaic, self.lrms, self.pan], f)

        print("Load data from cache file: ", cache_path)
        with open(cache_path, "rb") as f:
            self.mosaic, self.lrms, self.pan = pickle.load(f)

    def __len__(self):
        return len(self.lrms)

    def __getitem__(self, index):
        if self.mosaic != []:
            mosaic = torch.from_numpy(self.mosaic[index].astype(numpy.float32))
            mosaic = mosaic.permute(2, 0, 1)
            lrms = torch.from_numpy(self.lrms[index].astype(numpy.float32))
            lrms = lrms.permute(2, 0, 1)
            pan = torch.from_numpy(self.pan[index].astype(numpy.float32))
            pan = pan.permute(2, 0, 1)
            
            return mosaic, lrms, pan
        else:
            lrms, pan = self.lrms[index], self.pan[index]
            spatial_ratio = pan.shape[0] // lrms.shape[0]
            # for MS
            # blur kernel 
            lam1 = random.uniform(0.2, spatial_ratio)
            lam2 = random.uniform(lam1, spatial_ratio) if random.random() < 0.7 else lam1
            theta = random.uniform(0, numpy.pi)
            kernel, kernel_infos = shifted_anisotropic_Gaussian(k_size=7,
                                                                sf=2,
                                                                lambda_1=lam1**2,
                                                                lambda_2=lam2**2,
                                                                theta=theta,
                                                                shift=False)
            
            # blurring
            ms_blur = scipy.ndimage.convolve(lrms, kernel[:, :, numpy.newaxis], mode='reflect')
            ms_blur = numpy.clip(ms_blur, a_min=0.0, a_max=1.0)
            
            # downsampling
            ms_blur = cv2.resize(ms_blur, dsize=None, fx=1/spatial_ratio, fy=1/spatial_ratio, interpolation=cv2.INTER_CUBIC)

            # adding noise
            std = random.uniform(0.1, 15) / 255.0
            ms_lr = ms_blur + torch.randn(ms_blur.shape, dtype=torch.float32).numpy() * std
            ms_lr = numpy.clip(ms_lr, a_min=0, a_max=1.0)

            ms_label = torch.from_numpy(lrms.transpose([2,0,1])).type(torch.float32)        # c x h x w
            ms_lr = torch.from_numpy(ms_lr.transpose([2,0,1])).type(torch.float32)          # c x h x w
            ms_blur = torch.from_numpy(ms_blur.transpose([2,0,1])).type(torch.float32)      # c x h x w
            kernel_infos = torch.from_numpy(kernel_infos).type(torch.float32)               # 3
            nlevel = torch.tensor([std], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)   # 1 x 1 x 1

            # for PAN
            # downsampling
            pan_label_down = cv2.resize(pan, dsize=None, fx=1/spatial_ratio, fy=1/spatial_ratio, interpolation=cv2.INTER_CUBIC)[:, :, numpy.newaxis]
            pan_n = pan_label_down

            std = random.uniform(0.1, 30) / 255.0
            pan_n = pan_n + torch.randn(pan_n.shape, dtype=torch.float32).numpy() * std
            pan_n = numpy.clip(pan_n, a_min=0, a_max=1.0)

            pan_eps2 = 1e-5
            pan_sigma_spatial = 3
            pan_window_size = 2 * pan_sigma_spatial + 1
            sigma2_map_est = sigma_estimate(im_noisy=pan_n, im_gt=pan_label_down, win=pan_window_size,
                                            sigma_spatial=pan_sigma_spatial)
            tensor_pan_label = torch.from_numpy(pan_label_down.transpose([2,0,1])).type(torch.float32)
            tensor_pan_n = torch.from_numpy(pan_n.transpose([2,0,1])).type(torch.float32)
            tensor_map = torch.from_numpy(sigma2_map_est.transpose([2,0,1])).type(torch.float32)
            tensor_eps2 = torch.tensor([pan_eps2], dtype=torch.float32).reshape((1, 1, ))
            
            return ms_label, ms_lr, ms_blur, kernel_infos, nlevel, \
            tensor_pan_label, tensor_pan_n, tensor_map, tensor_eps2