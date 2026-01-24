import torch
import os
import numpy
import scipy.io as scio
import quality_index
import argparse
from tqdm.contrib import tzip
import tqdm
import utils
import h5py

def main(args):
    MSFA = numpy.array([[0, 1, 2, 3],
                        [4, 5, 6, 7],
                        [8, 9, 10, 11],
                        [12, 13, 14, 15]])
    msfa_kernel = torch.zeros(MSFA.shape[0] * MSFA.shape[1], 1, MSFA.shape[0]*2, MSFA.shape[1]*2)
    for i in range(MSFA.shape[0]):
        for j in range(MSFA.shape[1]):
            msfa_kernel[int(MSFA[i, j]), 0, i*2, j*2] = 0.25
            msfa_kernel[int(MSFA[i, j]), 0, i*2+1, j*2] = 0.25
            msfa_kernel[int(MSFA[i, j]), 0, i*2, j*2+1] = 0.25
            msfa_kernel[int(MSFA[i, j]), 0, i*2+1, j*2+1] = 0.25

    device = "cpu" if args.cpu == True else f"cuda:{args.device}"

    if args.simulate == True:
        mosaic_path = os.path.join(args.data_path, "mosaic")
        pan_path = os.path.join(args.data_path, "pan")
        fused_path = os.path.join(args.data_path, "fused")
        gt_path = os.path.join(args.data_path, "gt")

        mosaic_names = [file for file in os.listdir(mosaic_path) if file in args.data_id]
        pan_names = [file for file in os.listdir(pan_path) if file in args.data_id]
        fused_names = [file for file in os.listdir(fused_path) if file in args.data_id]
        gt_names = [file for file in os.listdir(gt_path) if file in args.data_id]
        
        if mosaic_names == []:
            mosaic_names = os.listdir(mosaic_path)
            pan_names = os.listdir(pan_path)
            fused_names = os.listdir(fused_path)
            gt_names = os.listdir(gt_path)
            print("To be tested: ALL. ", mosaic_names)
        mosaic_names.sort()
        pan_names.sort()
        fused_names.sort()
        gt_names.sort()

        psnr_avg, ssim_avg, sam_avg, ergas_avg, q2n_avg = 0, 0, 0, 0, 0
        for mosaic_name, pan_name, fused_name, gt_name in tqdm.tqdm(tzip(mosaic_names, pan_names, fused_names, gt_names)):
            mosaic = scio.loadmat(os.path.join(mosaic_path, mosaic_name))["mosaic"]
            pan = scio.loadmat(os.path.join(pan_path, pan_name))["pan"]
            fused = scio.loadmat(os.path.join(fused_path, fused_name))["fused"]
            gt = scio.loadmat(os.path.join(gt_path, gt_name))["gt"]

            mosaic_tensor = torch.from_numpy(mosaic.astype(numpy.float32)).permute(2, 0, 1).unsqueeze(0)
            mosaic_tensor = torch.nn.functional.pixel_shuffle(mosaic_tensor, upscale_factor=msfa_kernel.shape[2]//2)
            pan_tensor = torch.from_numpy(pan.astype(numpy.float32)).permute(2, 0, 1).unsqueeze(0)
            fused_tensor = torch.from_numpy(fused.astype(numpy.float32)).permute(2, 0, 1).unsqueeze(0)
            gt_tensor = torch.from_numpy(gt.astype(numpy.float32)).permute(2, 0, 1).unsqueeze(0)

            psnr_avg += quality_index.calc_psnr(gt_tensor, fused_tensor).item()
            ssim_avg += quality_index.calc_ssim(gt_tensor, fused_tensor).item()
            sam_avg += quality_index.calc_sam(gt_tensor, fused_tensor).item()
            ergas_avg += quality_index.calc_ergas(gt_tensor, fused_tensor).item()
            if gt_tensor.shape[-1] > 1000 and gt_tensor.shape[-2] > 1000:
                q2n_avg += quality_index.calc_q2n(gt_tensor[0], fused_tensor[0], Q_blocks_size=256, Q_shift=128)[0].item()
            else: 
                q2n_avg += quality_index.calc_q2n(gt_tensor[0], fused_tensor[0], Q_blocks_size=64, Q_shift=32)[0].item()
        psnr_avg /= len(mosaic_names)
        ssim_avg /= len(mosaic_names)
        sam_avg /= len(mosaic_names)
        ergas_avg /= len(mosaic_names)
        q2n_avg /= len(mosaic_names)
        
        print("PSNR: ", psnr_avg,
              "SSIM: ", ssim_avg,
              "SAM: ", sam_avg,
              "ERGAS: ", ergas_avg,
              "Q2n: ", q2n_avg,
              )
         
    elif args.real_world == True:
        mosaic_path = os.path.join(args.data_path, "mosaic")
        pan_path = os.path.join(args.data_path, "pan")
        fused_path = os.path.join(args.data_path, "fused")
        mosaic_names = os.listdir(mosaic_path)
        pan_names = os.listdir(pan_path)
        fused_names = os.listdir(fused_path)

        mosaic_path = os.path.join(args.data_path, "mosaic")
        pan_path = os.path.join(args.data_path, "pan")
        fused_path = os.path.join(args.data_path, "fused")

        mosaic_names = [file for file in os.listdir(mosaic_path) if file in args.data_id]
        pan_names = [file for file in os.listdir(pan_path) if file in args.data_id]
        fused_names = [file for file in os.listdir(fused_path) if file in args.data_id]
        if mosaic_names != []:
            print("To be tested: ", args.data_id)
        else:
            mosaic_names = os.listdir(mosaic_path)
            pan_names = os.listdir(pan_path)
            fused_names = os.listdir(fused_path)
            print("To be tested: ALL. ", mosaic_names)
        mosaic_names.sort()
        pan_names.sort()
        fused_names.sort()

        qnr_avg, d_lambda_avg, d_s_avg = 0, 0, 0
        for mosaic_name, pan_name, fused_name in tqdm.tqdm(tzip(mosaic_names, pan_names, fused_names)):
            mosaic = scio.loadmat(os.path.join(mosaic_path, mosaic_name))["mosaic"]
            pan = scio.loadmat(os.path.join(pan_path, pan_name))["pan"]
            fused = scio.loadmat(os.path.join(fused_path, fused_name))["fused"]

            mosaic_tensor = torch.from_numpy(mosaic.astype(numpy.float64)).permute(2, 0, 1).unsqueeze(0)
            mosaic_tensor = torch.nn.functional.pixel_shuffle(mosaic_tensor, upscale_factor=msfa_kernel.shape[2]//2)
            pan_tensor = torch.from_numpy(pan.astype(numpy.float64)).permute(2, 0, 1).unsqueeze(0)
            fused_tensor = torch.from_numpy(fused.astype(numpy.float64)).permute(2, 0, 1).unsqueeze(0)

            qnr, f_lambda, d_s = quality_index.calc_qnr_mosaic(fused_tensor.to(device), mosaic_tensor.to(device), pan_tensor.to(device), msfa_kernel.to(device), patch_size=args.patch_size, scale_factor=2)
            qnr_avg += qnr.item()
            d_lambda_avg += f_lambda.item()
            d_s_avg += d_s.item()

        qnr_avg /= len(mosaic_names)
        d_lambda_avg /= len(mosaic_names)
        d_s_avg /= len(mosaic_names)
        
        print("QNR: ", qnr_avg,
              "D_lambda: ", d_lambda_avg,
              "D_S: ", d_s_avg)

    return 
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--idx', type=int, default=1, help='Index to identify models.')
    parser.add_argument('--simulate', action='store_true', default=False, help='Determine whether to do simulating experiment')
    parser.add_argument('--real_world', action='store_true', default=False, help='Determine whether to do real-world experiment')
    parser.add_argument('--mosaic_save', action='store_true', default=False, help='Determine whether to generate mosaic data.')
    parser.add_argument('--pan_save', action='store_true', default=False, help='Determine whether to generate pan data.')
    parser.add_argument('--demosaic_save', action='store_true', default=False, help='Determine whether to generate demosaic data.')
    parser.add_argument('--gt_save', action='store_true', default=False, help='Determine whether to generate gt data. Effective only when in simulated dataset.')
    parser.add_argument('--spatial_ratio', type=int, default=2, help='Ratio of spatial resolutions between MS and PAN')
    parser.add_argument('--dataset', type=str, default="CAVE", help='Type of satellite data.')
    parser.add_argument('--num_bands', type=int, default=16, help='Number of bands of a MS image.')
    parser.add_argument('--msfa_size', type=int, default=4, help='Size of MSFA')
    parser.add_argument('--patch_size', type=int, default=16, help='Size of the patch when calculating QNR')
    parser.add_argument('--data_path', type=str, default="../DataSet/", help='Path of the dataset.')
    parser.add_argument('--data_id', type=str, default=[], nargs="+",
                        help='Index of which data to be tested. If empty, then all be selected.')
    parser.add_argument('--cpu', action='store_true', default=False, help='Determine whether to cpu.')
    parser.add_argument('--device', type=str, default='0', help='Device to train the model.')

    args = parser.parse_args()
    main(args)