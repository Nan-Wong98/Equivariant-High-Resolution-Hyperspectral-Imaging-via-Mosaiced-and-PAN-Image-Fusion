import numpy
import torch
import os
import matplotlib.pyplot as plt
import copy
from scipy.signal import convolve2d
import json

def pixel_shuffle(tensor, scale_factor):
    """
    Implementation of pixel shuffle using numpy

    Parameters:
    -----------
    tensor: input tensor, shape is [N, C, H, W]
    scale_factor: scale factor to up-sample tensor

    Returns:
    --------
    tensor: tensor after pixel shuffle, shape is [N, C/(s*s), s*H, s*W],
        where s refers to scale factor
    """
    num, ch, height, width = tensor.shape
    if ch % (scale_factor * scale_factor) != 0:
        raise ValueError('channel of tensor must be divisible by '
                         '(scale_factor * scale_factor).')

    new_ch = ch // (scale_factor * scale_factor)
    new_height = height * scale_factor
    new_width = width * scale_factor

    tensor = tensor.reshape(
        [num, new_ch, scale_factor, scale_factor, height, width])
    # new axis: [num, new_ch, height, scale_factor, width, scale_factor]
    tensor = tensor.transpose([0, 1, 4, 2, 5, 3])
    tensor = tensor.reshape([num, new_ch, new_height, new_width])
    return tensor


def pixel_shuffle_inv(tensor, scale_factor):
    """
    Implementation of inverted pixel shuffle using numpy

    Parameters:
    -----------
    tensor: input tensor, shape is [N, C, H, W]
    scale_factor: scale factor to down-sample tensor

    Returns:
    --------
    tensor: tensor after pixel shuffle, shape is [N, (s*s)*C, H/s, W/s],
        where s refers to scale factor
    """
    num, ch, height, width = tensor.shape
    if height % scale_factor != 0 or width % scale_factor != 0:
        raise ValueError('height and widht of tensor must be divisible by '
                         'scale_factor.')

    new_ch = ch * (scale_factor * scale_factor)
    new_height = height // scale_factor
    new_width = width // scale_factor

    tensor = tensor.reshape(
        [num, ch, new_height, scale_factor, new_width, scale_factor])
    # new axis: [num, ch, scale_factor, scale_factor, new_height, new_width]
    tensor = tensor.transpose([0, 1, 3, 5, 2, 4])
    tensor = tensor.reshape([num, new_ch, new_height, new_width])
    return tensor

def compress(array):
    compress_data = numpy.empty_like(array)

    def cumulativehistogram(array, band_min, band_max):
        gray_level = 100
        interval = (band_max - band_min) / gray_level

        # Histogram computation using numpy.bincount
        histogram = numpy.empty(gray_level)
        for i in range(gray_level):
            histogram[i] = numpy.sum(numpy.logical_and(array >= band_min + i*interval, array < (band_min + (i+1)*interval)))
        cumulative_histogram = numpy.cumsum(histogram)

        count_percent2 = array.size * 0.02
        count_percent98 = array.size * 0.98

        # Find cutmin and cutmax using numpy.argmax with conditions
        cutmin_index = numpy.argmax(cumulative_histogram >= count_percent2)*interval + band_min
        cutmax_index = numpy.argmax(cumulative_histogram >= count_percent98)*interval + band_min
        return cutmin_index, cutmax_index

    for i in range(array.shape[-1]):
        band_max = numpy.max(array[:, :, i])
        band_min = numpy.min(array[:, :, i])

        cutmin_index, cutmax_index = cumulativehistogram(array[:, :, i], band_min, band_max)

        # Vectorized operation to compute compress_data
        compress_data[:, :, i] = ((numpy.clip(array[:, :, i], cutmin_index, cutmax_index) - cutmin_index) / (
                (cutmax_index - cutmin_index) / 255)).astype(numpy.uint8)

    return compress_data

def plt_diff(diff_map, save_path, upper_boundary=64):
    h, w = diff_map.shape[:2]
    fig = plt.figure(figsize=(w/100, h/100), dpi=100, frameon=False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(
        diff_map, 
        cmap=plt.get_cmap('viridis', 256), 
        vmin=0, 
        vmax=upper_boundary,
        extent=[0, w, 0, h]
    )
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(
        save_path, 
        dpi=100,
        bbox_inches='tight', 
        pad_inches=0
    )
    plt.close()
    return


def plt_colorbar(save_path):
    plt.figure()
    im = plt.imshow(numpy.tile(numpy.arange(64), (10, 1)))

    fig_colorbar = plt.figure(figsize=(1, 10))
    plt.colorbar(im, cax=plt.gca(), ax=None)
    plt.xticks([])
    plt.yticks([])
    fig_colorbar.savefig(save_path, bbox_inches='tight', pad_inches=0)
    return


def detach_(data, size, coordinate, box_width, rgb):
    data[coordinate[0]: coordinate[0] + box_width, coordinate[1]: coordinate[1] + size[1]] = 0
    data[coordinate[0] + size[0] - box_width: coordinate[0] + size[0], coordinate[1]: coordinate[1] + size[1]] = 0
    data[coordinate[0]: coordinate[0] + size[0], coordinate[1]: coordinate[1] + box_width] = 0
    data[coordinate[0]: coordinate[0] + size[0], coordinate[1] + size[1] - box_width: coordinate[1] + size[1]] = 0

    data[coordinate[0]: coordinate[0] + box_width, coordinate[1]: coordinate[1] + size[1], rgb] = 255
    data[coordinate[0] + size[0] - box_width: coordinate[0] + size[0], coordinate[1]: coordinate[1] + size[1],
    rgb] = 255
    data[coordinate[0]: coordinate[0] + size[0], coordinate[1]: coordinate[1] + box_width, rgb] = 255
    data[coordinate[0]: coordinate[0] + size[0], coordinate[1] + size[1] - box_width: coordinate[1] + size[1],
    rgb] = 255

    local_region = copy.deepcopy(data[coordinate[0]: coordinate[0] + size[0], coordinate[1]: coordinate[1] + size[1]])

    return local_region, data


def region_detach(data, size, coordinate, box_width=5, rgb="r"):
    if len(size) > 2 or len(size) == 0:
        raise ValueError(f"`zoomin_size` out of length: {len(size)}")
    elif len(size) == 1:
        size = (size[0], size[0])

    if len(coordinate) > 4 or len(coordinate) == 0 or len(coordinate) == 3 or len(coordinate) == 1:
        raise ValueError(f"`coordinate` out of length: {len(size)}")
    elif len(coordinate) == 2:
        coordinate = (coordinate[0], coordinate[0], coordinate[1], coordinate[1])

    if coordinate[0] + size[0] > data.shape[0] or coordinate[1] + size[1] > data.shape[1]:
        raise RuntimeError(
            f"Detached region out of bound. Coordinate: {coordinate}, size: {size}, bound: {data.shape[:2]}")

    if coordinate[2] + size[0] > data.shape[0] or coordinate[3] + size[1] > data.shape[1]:
        raise RuntimeError(
            f"Detached region out of bound. Coordinate: {coordinate}, size: {size}, bound: {data.shape[:2]}")

    if rgb != "r" and rgb != "b":
        raise ValueError(f"Incorrect boxcolor: {rgb}. It should be r or b. ")

    rgb = 2 if rgb == "r" else 0
    local_region1, data_with_box1 = detach_(data, size, coordinate[:2], box_width, rgb=rgb)
    local_region2, data_with_box2 = detach_(data_with_box1, size, coordinate[2:], box_width, rgb=2 - rgb)

    return local_region1, local_region2, data_with_box2, size, coordinate

def MSFA_filter(data, MSFA_k):
    H, W, B = data.shape
    assert B == MSFA_k.shape[0] * MSFA_k.shape[1], f"Number of bands: {B}. Shape of MSFA: {MSFA_k.shape}"
    MSFA_k_h, MSFA_k_w = MSFA_k.shape
    assert H % MSFA_k_h == 0 and W % MSFA_k_w == 0, f"Height or width of data: {H}, {W} is not divided by height or width of blur kernel: {MSFA_k_h}, {MSFA_k_w}"
    
    MSFA_kernel = numpy.zeros((MSFA_k_h, MSFA_k_w, B))
    for i in range(MSFA_k_h):
        for j in range(MSFA_k_w):
            MSFA_kernel[i, j, int(MSFA_k[i, j])] = 1
    MSFA_map = numpy.tile(MSFA_kernel, (H // MSFA_k_h, W // MSFA_k_w, 1))
    mosaic = numpy.sum(data * MSFA_map, axis=-1, keepdims=True)
    # mosaic = pixel_shuffle_inv(mosaic.transpose(2, 0, 1)[numpy.newaxis,:], 4)[0].transpose(1, 2, 0)
    return mosaic

def MSFA_filter_inv(data, MSFA_k):
    H, W, _ = data.shape
    MSFA_k_h, MSFA_k_w = MSFA_k.shape
    assert H % MSFA_k_h == 0 and W % MSFA_k_w == 0, f"Height or width of data: {H}, {W} is not divided by height or width of blur kernel: {MSFA_k_h}, {MSFA_k_w}"
    B = MSFA_k_h * MSFA_k_w

    mosaic_inv = numpy.zeros((H // MSFA_k_h, W // MSFA_k_w, B))
    for i in range(MSFA_k_h):
        for j in range(MSFA_k_w):
            mosaic_inv[:, :, int(MSFA_k[i, j])] = data[i::MSFA_k_h, j::MSFA_k_w, 0]

    return mosaic_inv

def gaussian_kernel(ks=5, sigma=3):
    kernel = numpy.zeros((ks, ks))
    radius = ks // 2
    for y in range(-radius, radius + 1):
        for x in range(-radius, radius + 1):
            v = 1.0 / (2 * numpy.pi * sigma ** 2) * numpy.exp(-1.0 / (2 * sigma ** 2) * (x ** 2 + y ** 2))
            kernel[y + radius, x + radius] = v
    kernel_ = kernel / numpy.sum(kernel)
    return kernel_

def blur_downsample(data, blur_k, scale_factor=2):
    H, W, B = data.shape
    assert H % scale_factor == 0 and W % scale_factor == 0, f"H: {H} or W: {W} is not divided by scale_factor: {scale_factor} "
    blur_k_h, blur_k_w = blur_k.shape

    data_torch = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0)
    blur_k_torch = torch.from_numpy(blur_k).unsqueeze(0).unsqueeze(0).expand(B, 1, blur_k.shape[0], blur_k.shape[1])
    data_blur = torch.nn.functional.conv2d(data_torch, blur_k_torch, padding=blur_k_h//2, groups=B)
    downsample_k = torch.ones(B, 1, scale_factor, scale_factor).type(torch.float32) / scale_factor**2
    data_downsample = torch.nn.functional.conv2d(data_blur, downsample_k, stride=scale_factor, groups=B)
    return data_downsample[0].permute(1, 2, 0).numpy()

def Lrms_and_Pan_simulate(raw_data, spe_res, MSFA_k, blur_k, scale_factor=2):
    B = raw_data.shape[-1]
    assert B == spe_res.shape[-1], f"Number of bands: {B} != last dimension of spectral response: {spe_res.shape[1]}"
    
    pan_sim = numpy.sum(raw_data * spe_res, axis=-1, keepdims=True)

    downsample = blur_downsample(raw_data, blur_k, scale_factor)

    lrms_sim = MSFA_filter(downsample, MSFA_k)

    return lrms_sim, pan_sim

def WB(mosaic, MSFA, F):
    mosaic_sparse = numpy.zeros((mosaic.shape[0], mosaic.shape[1], MSFA.shape[0]*MSFA.shape[1]))
    for i in range(MSFA.shape[0]):
        for j in range(MSFA.shape[1]):
            mosaic_sparse[i::MSFA.shape[0], j::MSFA.shape[1], i*MSFA.shape[1]+j] = mosaic[i::MSFA.shape[0], j::MSFA.shape[1], 0]

    demosaic_WB = numpy.zeros_like(mosaic_sparse)
    for c in range(demosaic_WB.shape[2]):
        demosaic_WB[:, :, c] = convolve2d(mosaic_sparse[:, :, c], F, mode='same', boundary='fill', fillvalue=0)
    
    return demosaic_WB

def plot_psnr_curve(load_path):
    record_file = [file for file in os.listdir(load_path) if file.endswith(".json") == True]
    if len(record_file) == 0:
        raise RuntimeError("There is no .json file in this directory: {}".format(load_path))
    record_file = record_file[0]
    with open(os.path.join(load_path, record_file), "r") as f:
        record = json.load(f)
        epoch, psnr = [], []
        for dt in record:
            epoch.append(dt["epoch"])
            psnr.append(dt["psnr"])
    plt.plot(epoch, psnr)
    plt.savefig(os.path.join(load_path, record_file.split(".")[0]+".png"))
    plt.close()

    return