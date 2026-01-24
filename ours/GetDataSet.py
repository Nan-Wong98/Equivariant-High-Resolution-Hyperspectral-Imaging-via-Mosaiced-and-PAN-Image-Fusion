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
import scipy
import math, random

def crop_to_patch(img, size, stride):
    H, W = img.shape[:2]
    patches = []
    for h in range(0, H, stride):
        for w in range(0, W, stride):
            if h + size <= H and w + size <= W:
                patch = img[h: h + size, w: w + size, :]
                patches.append(patch)
    return patches

def crop_to_patch_4d(img, size, stride):
    H, W = img.shape[1:3]
    patches = []
    for h in range(0, H, stride):
        for w in range(0, W, stride):
            if h + size <= H and w + size <= W:
                patch = img[:, h: h + size, w: w + size, :]
                for f in range(patch.shape[0]):
                    patches.append(patch[f])
    return patches

class MakeSimulateDataset(Dataset):
    def __init__(self, args, type="train"):
        cache_path = os.path.join(args.cache_path, type + "_cache.pkl")
        if not os.path.exists(cache_path):
            self.mosaic, self.pan, self.hrms = [], [], []
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

                # PAN simulate
                spe_res = numpy.array([1., 1, 2, 4, 8, 9, 10, 12, 16, 12, 10, 9, 7, 3, 2, 1])
                spe_res /= spe_res.sum()
                pan = numpy.sum(hrms_select_bands * spe_res, axis=-1, keepdims=True)
                
                spatial_ratio = pan.shape[0] // mosaic.shape[0]
                if type == "train":
                    mosaic_patches = crop_to_patch(mosaic, args.train_size//spatial_ratio, args.stride//spatial_ratio)
                    pan_patches = crop_to_patch(pan, args.train_size, args.stride)

                    self.mosaic += mosaic_patches
                    self.pan += pan_patches

                elif type == "test":
                    self.mosaic.append(mosaic)
                    self.pan.append(pan)
                    self.hrms.append(hrms_select_bands)

            with open(cache_path, "wb") as f:
                pickle.dump([self.mosaic, self.pan, self.hrms], f)

        print("Load data from cache file: ", cache_path)
        with open(cache_path, "rb") as f:
            self.mosaic, self.pan, self.hrms = pickle.load(f)

    def __len__(self):
        return len(self.mosaic)

    def __getitem__(self, index):
        if self.hrms != []:
            mosaic = torch.from_numpy(self.mosaic[index].astype(numpy.float32))
            mosaic = mosaic.permute(2, 0, 1)
            pan = torch.from_numpy(self.pan[index].astype(numpy.float32))
            pan = pan.permute(2, 0, 1)
            hrms = torch.from_numpy(self.hrms[index].astype(numpy.float32))
            hrms = hrms.permute(2, 0, 1)

            return mosaic, pan, hrms
        else:
            mosaic = torch.from_numpy(self.mosaic[index].astype(numpy.float32))
            mosaic = mosaic.permute(2, 0, 1)
            pan = torch.from_numpy(self.pan[index].astype(numpy.float32))
            pan = pan.permute(2, 0, 1)

            return mosaic, pan

class MakeRealWorldDataset(Dataset):
    def __init__(self, args, type="train"):
        cache_path = os.path.join(args.cache_path, type + "_cache.pkl")
        if not os.path.exists(cache_path):
            self.mosaic, self.pan = [], []
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
                if args.dataset == "Ours_mat":
                    mosaic = h5py.File(os.path.join(mosaic_path, mosaic_name))["msi"][:][numpy.newaxis, :, :, :]
                    mosaic = utils.pixel_shuffle(mosaic, args.msfa_size)[0].transpose(1, 2, 0)
                    pan = h5py.File(os.path.join(pan_path, pan_name))["pan"][:].transpose(1, 2, 0)
                elif args.dataset == "real_world":
                    with open(os.path.join(mosaic_path, mosaic_name), "rb") as f:
                        mosaic_raw = numpy.frombuffer(f.read(), dtype=numpy.int16)
                        mosaic = numpy.zeros((255*args.msfa_size, 276*args.msfa_size, 1))
                        for i in range(args.msfa_size):
                            for j in range(args.msfa_size):
                                mosaic[i::args.msfa_size, j::args.msfa_size, 0] = mosaic_raw[255*276*(i*args.msfa_size+j): 255*276*(i*args.msfa_size+j+1)].reshape(255, 276)
                    with open(os.path.join(pan_path, pan_name), "rb") as f:
                        pan_raw = numpy.frombuffer(f.read(), dtype=numpy.int16)
                        pan = pan_raw.reshape((2040, 2208, 1))
                assert pan.shape[1] % mosaic.shape[1] == 0 and pan.shape[2] % mosaic.shape[
                    2] == 0, "Mismatch between the spatial size of MS and PAN"
                spatial_ratio = pan.shape[1] // mosaic.shape[1]
                
                mosaic = mosaic / 2**12
                pan = pan / 2**12

                if type == "train":
                    mosaic_patches = crop_to_patch(mosaic, args.train_size // spatial_ratio, args.stride // spatial_ratio)
                    pan_patches = crop_to_patch(pan, args.train_size, args.stride)

                    self.mosaic += mosaic_patches
                    self.pan += pan_patches

                elif type == "test":
                    self.mosaic.append(mosaic)
                    self.pan.append(pan)

            with open(cache_path, "wb") as f:
                pickle.dump([self.mosaic, self.pan], f)

        print("Load data from cache file: ", cache_path)
        with open(cache_path, "rb") as f:
            self.mosaic, self.pan = pickle.load(f)

    def __len__(self):
        return len(self.mosaic)

    def __getitem__(self, index):
        mosaic = torch.from_numpy(self.mosaic[index].astype(numpy.float32))
        mosaic = mosaic.permute(2, 0, 1)
        pan = torch.from_numpy(self.pan[index].astype(numpy.float32))
        pan = pan.permute(2, 0, 1)

        return mosaic, pan

class MakeRealWorldVideoDataset(Dataset):
    def __init__(self, args, type="train"):
        cache_path = os.path.join(args.cache_path, type + "_cache.pkl")
        if not os.path.exists(cache_path):
            self.mosaic, self.pan = [], []
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
                with open(os.path.join(pan_path, pan_name), "rb") as f:
                    pan_raw = numpy.frombuffer(f.read(), dtype=numpy.int16)
                total_frames = len(mosaic_raw) // 255 // 276 // args.msfa_size**2
                mosaic = numpy.zeros((total_frames//args.frames, 255*args.msfa_size, 276*args.msfa_size, 1))
                pan = numpy.zeros((total_frames//args.frames, 255*args.msfa_size*2, 276*args.msfa_size*2, 1))
                for f in range(0, total_frames//args.frames*args.frames, args.frames):
                    for i in range(args.msfa_size):
                        for j in range(args.msfa_size):
                            mosaic[f//args.frames, i::args.msfa_size, j::args.msfa_size, 0] = mosaic_raw[f*255*276*args.msfa_size**2+255*276*(i*args.msfa_size+j): f*255*276*args.msfa_size**2+255*276*(i*args.msfa_size+j+1)].reshape(255, 276)
                    pan[f//args.frames] = pan_raw[f*255*276*args.msfa_size**2*4:(f+1)*255*276*args.msfa_size**2*4].reshape((2040, 2208, 1))

                spatial_ratio = pan.shape[1] // mosaic.shape[1]
                
                mosaic = mosaic / 2**12
                pan = pan / 2**12

                if type == "train":
                    mosaic_patches = crop_to_patch_4d(mosaic, args.train_size // spatial_ratio, args.stride // spatial_ratio)
                    pan_patches = crop_to_patch_4d(pan, args.train_size, args.stride)

                    self.mosaic += mosaic_patches
                    self.pan += pan_patches

                elif type == "test":
                    for f in range(mosaic.shape[0]):
                        self.mosaic.append(mosaic[f])
                        self.pan.append(pan[f])

            with open(cache_path, "wb") as f:
                pickle.dump([self.mosaic, self.pan], f)

        print("Load data from cache file: ", cache_path)
        with open(cache_path, "rb") as f:
            self.mosaic, self.pan = pickle.load(f)

    def __len__(self):
        return len(self.mosaic)

    def __getitem__(self, index):
        mosaic = torch.from_numpy(self.mosaic[index].astype(numpy.float32))
        mosaic = mosaic.permute(2, 0, 1)
        pan = torch.from_numpy(self.pan[index].astype(numpy.float32))
        pan = pan.permute(2, 0, 1)

        return mosaic, pan