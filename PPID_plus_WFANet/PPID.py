import numpy
import scipy.io as scio
import cv2
import os
import utils
from scipy.signal import convolve2d

def PPID(mosaic, MSFA, gamma_flag=False):
    '''
    From the paper, "Multispectral Demosaicing Using Pseudo-Panchromatic Image", 2017, Mihoubi et al
    An implementation with Numpy package.
    mosaic: numpy.ndarray (H, W), the mosaiced image
    MSFA: numpy.ndarray (h, w), the multi-spectral filter array
    gamma_flag: boolean, default to False, if `True`, interpolate the difference with the computed gamma map
        as the paper does. If `False`, interpolate just with the H map. When it is set to `False`, the demosaiced
        image will be better. Therefore, we are not sure whether we reproduce the PPID correctly and hope to see
        others help refine this function.
    '''
    # Eq. (21)
    lrms_mosaic_adjust_illu = mosaic
    for i in range(MSFA.shape[0]):
        for j in range(MSFA.shape[1]):
            lrms_mosaic_adjust_illu[i::MSFA.shape[0], j::MSFA.shape[1]] = lrms_mosaic_adjust_illu[i::MSFA.shape[0], j::MSFA.shape[1]]# * numpy.max(mosaic) / numpy.max(mosaic[i::MSFA.shape[0], j::MSFA.shape[1]])

    # Eq. (23)
    avg_filter = numpy.array([[1, 2, 2, 2, 1],
                            [2, 4, 4, 4, 2],
                            [2, 4, 4, 4, 2],
                            [2, 4, 4, 4, 2],
                            [1, 2, 2, 2, 1]]) / 64.

    # Eq. (24) FLOPs: H*W*5*5
    ppi_first = convolve2d(lrms_mosaic_adjust_illu[:, :, 0], avg_filter, mode='same', boundary='fill', fillvalue=0)[:, :, numpy.newaxis]

    ppi_first_pad = numpy.zeros((ppi_first.shape[0]+2*MSFA.shape[0],
                                ppi_first.shape[1]+2*MSFA.shape[1],
                                ppi_first.shape[2]))
    ppi_first_pad[MSFA.shape[0]:-MSFA.shape[0], MSFA.shape[1]:-MSFA.shape[1], :] = ppi_first

    lrms_mosaic_adjust_illu_pad = numpy.zeros((lrms_mosaic_adjust_illu.shape[0]+2*MSFA.shape[0]+4,
                                            lrms_mosaic_adjust_illu.shape[1]+2*MSFA.shape[1]+4,
                                            lrms_mosaic_adjust_illu.shape[2]))
    lrms_mosaic_adjust_illu_pad[MSFA.shape[0]+2:-MSFA.shape[0]-2, MSFA.shape[1]+2:-MSFA.shape[1]-2, :] = lrms_mosaic_adjust_illu

    ppi = numpy.zeros_like(lrms_mosaic_adjust_illu)
    gamma_map = numpy.zeros((lrms_mosaic_adjust_illu.shape[0], lrms_mosaic_adjust_illu.shape[1], 9))

    # FLOPs: 8 * (6*H*W+H*W)
    numerator_of_eq28 = numpy.zeros_like(lrms_mosaic_adjust_illu)
    denominator_of_eq28 = numpy.zeros_like(lrms_mosaic_adjust_illu)
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i==0 and j==0:
                continue
            gamma_q = numpy.ones_like(lrms_mosaic_adjust_illu)
            # FLOPs: 6 * H*W
            for v in range(-1, 2):
                for u in range(0, 2):
                    # Eq. (26)
                    if i*j == 0:
                        h_offset = i*u + j*v
                        w_offset = j*u + i*v
                    else:
                        h_offset = (u+abs(v)*(1-v)/2)*i
                        w_offset = (u+abs(v)*(1+v)/2)*j
                    h_offset = int(h_offset)
                    w_offset = int(w_offset)
                    # 0, i=-1, j=-1 -> q1
                    # 1, i=-1, j=0  -> q2
                    # 2, i=-1, j=1  -> q3
                    # 3, i=0, j=-1  -> q8
                    # 4, i=0, j=0   -> p
                    # 5, i=0, j=1   -> q4
                    # 6, i=1, j=-1  -> q7
                    # 7, i=1, j=0   -> q6
                    # 8, i=1, j=1   -> q5
                    lrms_mosaic_adjust_illu_pad_offset = numpy.zeros_like(lrms_mosaic_adjust_illu_pad)
                    H, W = lrms_mosaic_adjust_illu_pad_offset.shape[:2]
                    lrms_mosaic_adjust_illu_pad_offset[(1+i)*MSFA.shape[0]+2+h_offset:H+(i-1)*MSFA.shape[0]-2+h_offset, (1+j)*MSFA.shape[1]+2+w_offset:W+(j-1)*MSFA.shape[1]-2+w_offset, :] = lrms_mosaic_adjust_illu

                    # Eq. (25) FLOPs: H*W
                    gamma_q += (2-u)*(2-abs(v))*abs(lrms_mosaic_adjust_illu_pad - lrms_mosaic_adjust_illu_pad_offset)[MSFA.shape[0]+2:-MSFA.shape[0]-2, MSFA.shape[1]+2:-MSFA.shape[1]-2, :]
            gamma_q = 1. / gamma_q
            gamma_map[:, :, (i+1)*3+j+1] = gamma_q[:, :, -1]
            ppi_first_pad_tmp = numpy.zeros_like(ppi_first_pad)
            H, W = ppi_first_pad_tmp.shape[:2]
            ppi_first_pad_tmp[(1+i)*MSFA.shape[0]:H+(i-1)*MSFA.shape[0], (1+j)*MSFA.shape[1]:W+(j-1)*MSFA.shape[1], :] = ppi_first
            lrms_mosaic_adjust_illu_pad_offset_tmp = numpy.zeros_like(ppi_first_pad)
            H, W = lrms_mosaic_adjust_illu_pad_offset_tmp.shape[:2]
            lrms_mosaic_adjust_illu_pad_offset_tmp[(1+i)*MSFA.shape[0]:H+(i-1)*MSFA.shape[0], (1+j)*MSFA.shape[1]:W+(j-1)*MSFA.shape[1], :] = lrms_mosaic_adjust_illu

            # FLOPs: H*W
            numerator_of_eq28 += gamma_q * (ppi_first_pad_tmp - lrms_mosaic_adjust_illu_pad_offset_tmp)[MSFA.shape[0]:-MSFA.shape[0], MSFA.shape[1]:-MSFA.shape[1], :]
            denominator_of_eq28 += gamma_q
    # Eq. (28) FLOPs: H*W
    ppi = lrms_mosaic_adjust_illu + numerator_of_eq28 / denominator_of_eq28

    F = numpy.array([[1, 2, 3, 4, 3, 2, 1],
                    [2, 4, 6, 8, 6, 4, 2],
                    [3, 6, 9, 12, 9, 6, 3],
                    [4, 8, 12, 16, 12, 8, 4],
                    [3, 6, 9, 12, 9, 6, 3],
                    [2, 4, 6, 8, 6, 4, 2],
                    [1, 2, 3, 4, 3, 2, 1]])/16.
    # FLOPs: 16 * H*W*7*7
    diff_estimate = numpy.zeros((ppi.shape[0], ppi.shape[1], MSFA.shape[0]*MSFA.shape[1]))
    for i in range(MSFA.shape[0]):
        for j in range(MSFA.shape[1]):
            lrms_mosaic_adjust_illu_sparse = numpy.zeros_like(ppi)
            ppi_sparse = numpy.zeros_like(ppi)
            lrms_mosaic_adjust_illu_sparse[i::MSFA.shape[0], j::MSFA.shape[1]] = lrms_mosaic_adjust_illu[i::MSFA.shape[0], j::MSFA.shape[1]]
            ppi_sparse[i::MSFA.shape[0], j::MSFA.shape[1]] = ppi[i::MSFA.shape[0], j::MSFA.shape[1]]
            # Eq. (30)
            diff_tmp = lrms_mosaic_adjust_illu_sparse[:, :, -1] - ppi_sparse[:, :, -1]
            if gamma_flag == False:
                # Eq. (31) FLOPs: H*W*7*7
                diff_estimate[:, :, i*MSFA.shape[1]+j] = convolve2d(diff_tmp, F, mode='same', boundary='fill', fillvalue=0)
            else:
                diff_tmp_pad = numpy.zeros((diff_tmp.shape[0]+2*F.shape[0]//2, diff_tmp.shape[1]+2*F.shape[1]//2))
                diff_tmp_pad[F.shape[0]//2: -F.shape[0]//2, F.shape[0]//2: -F.shape[0]//2] = diff_tmp
                # Eq. (32)
                for h in range(diff_estimate.shape[0]):
                    for w in range(diff_estimate.shape[1]):
                        # Eq. (33)
                        gamma_filter = numpy.zeros_like(F)
                        gamma_filter[:3, :3] = gamma_map[h, w, 0]    # q1
                        gamma_filter[:3, 4:] = gamma_map[h, w, 2]    # q3
                        gamma_filter[4:, :3] = gamma_map[h, w, 6]    # q7
                        gamma_filter[4:, 4:] = gamma_map[h, w, 8]    # q5
                        gamma_filter[3, :3] = gamma_map[h, w, 3]     # q8
                        gamma_filter[3, 4:] = gamma_map[h, w, 5]     # q4
                        gamma_filter[:3, 3] = gamma_map[h, w, 1]     # q2
                        gamma_filter[4:, 3] = gamma_map[h, w, 7]     # q6
                        gamma_filter[3, 3] = 1
                        # Eq. (31)
                        diff_estimate[h, w, i*MSFA.shape[1]+j] = numpy.sum(diff_tmp_pad[h:h+F.shape[0], w:w+F.shape[1]] * gamma_filter * F)
    # Eq. (34)
    demosaic_estimate = diff_estimate + ppi

    # Eq. (22)
    for i in range(MSFA.shape[0]):
        for j in range(MSFA.shape[1]):
            demosaic_estimate[i::MSFA.shape[0], j::MSFA.shape[1]] = demosaic_estimate[i::MSFA.shape[0], j::MSFA.shape[1]]# / numpy.max(mosaic) * numpy.max(mosaic[i::MSFA.shape[0], j::MSFA.shape[1]])

    # All FLOPs: 883*H*W
    return demosaic_estimate

if __name__=="__main__":
    hrms = scio.loadmat("../DataSet/CAVE/test/egyptian_statue_ms.mat")["b"]

    hrms_select_bands = hrms[:, :, 12:28]

    MSFA = numpy.array([[0, 1, 2, 3],
                        [4, 5, 6, 7],
                        [8, 9, 10, 11],
                        [12, 13, 14, 15]])

    blur_k = utils.gaussian_kernel(5, 3)

    lrms = utils.blur_downsample(hrms_select_bands, blur_k, scale_factor=2)

    lrms_mosaic = utils.MSFA_filter(lrms, MSFA)

    spe_res = numpy.ones(hrms_select_bands.shape[-1]) / hrms_select_bands.shape[-1]
    pan = numpy.sum(hrms_select_bands * spe_res, axis=-1, keepdims=True)
    ppi_avg = numpy.mean(lrms, axis=-1, keepdims=True)

    demosaic_estimate = PPID(lrms_mosaic, MSFA)


