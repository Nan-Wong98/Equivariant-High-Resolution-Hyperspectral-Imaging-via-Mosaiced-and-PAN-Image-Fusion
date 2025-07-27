from model import SFNet, Generator
import torch
import os
import numpy
import scipy.io as scio
import cv2
import argparse
from tqdm.contrib import tzip
import tqdm
import utils
import h5py

def main(args):
    dir_dataset = os.path.join("./", args.dataset)
    if not os.path.exists(dir_dataset):
        os.mkdir(dir_dataset)

    dir_idx = os.path.join(dir_dataset, str(args.idx))
    if not os.path.exists(dir_idx):
        os.mkdir(dir_idx)
    
    device = "cpu" if args.cpu == True else f"cuda:{args.device}"

    demosaic_net, ps_net = SFNet(args), Generator(args)
    if args.load_ps_model == "" or args.load_ps_model == None:
        print("==> No demosaicing_net checkpoint loaded!")
    else:
        print("==> Load demosaicing_net checkpoint: {}".format(args.load_demosaic_model))
        demosaic_net.load_state_dict(torch.load(args.load_demosaic_model, map_location="cpu"), strict=False)
        print("==> Load pansharpening_net checkpoint: {}".format(args.load_ps_model))
        ps_net.load_state_dict(torch.load(args.load_ps_model, map_location="cpu")["psnet"], strict=False)

    ps_net = ps_net.to(device)
    ps_net.eval()
    demosaic_net = demosaic_net.to(device)
    demosaic_net.eval()

    data_path = os.path.join(args.data_path, args.dataset, "test")
    ids, mosaics, pans, gts = [], [], [], []

    if args.simulate == True:
        if args.data_id == []:
            data_names = os.listdir(data_path)
        else:
            data_names = [file for file in os.listdir(data_path) if file in args.data_id]
        MSFA = numpy.array([[0, 1, 2, 3],
                            [4, 5, 6, 7],
                            [8, 9, 10, 11],
                            [12, 13, 14, 15]])

        for data_name in tqdm.tqdm(data_names):
            data_dir = os.path.join(data_path, data_name)
            if args.dataset == "CAVE":
                hrms = scio.loadmat(data_dir)['b']
            elif args.dataset == "ICVL":
                hrms = h5py.File(data_dir)["rad"][:]
                hrms = numpy.rot90(hrms.transpose(2, 1, 0))
                hrms /= hrms.max((0, 1))
            hrms_select_bands = hrms[:hrms.shape[0]//(args.msfa_size*args.spatial_ratio)*(args.msfa_size*args.spatial_ratio),
                                    :hrms.shape[1]//(args.msfa_size*args.spatial_ratio)*(args.msfa_size*args.spatial_ratio),
                                    12:28].astype(numpy.float32)
        
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

            mosaics.append(mosaic)
            pans.append(pan)
            gts.append(hrms_select_bands)
            ids.append(data_name.split(".")[0])

    if args.simulate == False:
        mosaic_path = os.path.join(data_path, "mosaic")
        pan_path = os.path.join(data_path, "pan")
        if args.data_id == []:
            mosaic_imgs = os.listdir(mosaic_path)
            pan_imgs = os.listdir(pan_path)
        else:
            mosaic_imgs = [file for file in os.listdir(mosaic_path) if file in args.data_id]
            pan_imgs = [file for file in os.listdir(pan_path) if file in args.data_id]
        mosaic_imgs.sort()
        pan_imgs.sort()
        assert len(mosaic_imgs) == len(pan_imgs), "Length mismatch between MS and PAN images!"

        for mosaic_name, pan_name in tqdm.tqdm(tzip(mosaic_imgs, pan_imgs)):
            with open(os.path.join(mosaic_path, mosaic_name), "rb") as f:
                mosaic_raw = numpy.frombuffer(f.read(), dtype=numpy.int16)
                mosaic = numpy.zeros((255*4, 276*4, 1)).astype(numpy.int16)
                for i in range(args.msfa_size):
                    for j in range(args.msfa_size):
                        mosaic[i::args.msfa_size, j::args.msfa_size, 0] = mosaic_raw[255*276*(i*args.msfa_size+j): 255*276*(i*args.msfa_size+j+1)].reshape(255, 276)
                mosaics.append(mosaic)
            with open(os.path.join(pan_path, pan_name), "rb") as f:
                pan_raw = numpy.frombuffer(f.read(), dtype=numpy.int16)
                pan = pan_raw.reshape((255*8, 276*8, 1))
                pans.append(pan)
            ids.append(mosaic_name.split(".")[0])

    dir_mat = os.path.join(dir_idx, "result", "mat")
    if not os.path.exists(dir_mat):
        os.makedirs(dir_mat)
    print("Start to generate the real_world results!")

    MSFA = numpy.array([[0, 1, 2, 3],
                        [4, 5, 6, 7],
                        [8, 9, 10, 11],
                        [12, 13, 14, 15]])
    for cnt, (idx, mosaic, pan) in enumerate(tqdm.tqdm(tzip(ids, mosaics, pans))):
        if args.simulate == False:
            mosaic_tensor = torch.from_numpy(mosaic.astype(numpy.float32)/2**12).permute(2, 0, 1).unsqueeze(0).to(device)
            pan_tensor = torch.from_numpy(pan.astype(numpy.float32)/2**12).permute(2, 0, 1).unsqueeze(0).to(device)
        else:
            mosaic_tensor = torch.from_numpy(mosaic.astype(numpy.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
            pan_tensor = torch.from_numpy(pan.astype(numpy.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            demosaic_tensor = demosaic_net(mosaic_tensor)
            demosaic = demosaic_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
            
            hrms_tensor = ps_net(demosaic_tensor, pan_tensor)
            if args.simulate == False:
                hrms_tensor = torch.clamp(torch.round(hrms_tensor*2**12), 0, 2**12).short()
            hrms_tensor = hrms_tensor.detach()[0].cpu()
        
        hrms = hrms_tensor.permute(1, 2, 0).numpy()

        if args.mosaic_save == True:
            mosaic_numpy = numpy.zeros((mosaic.shape[0]//MSFA.shape[0], mosaic.shape[1]//MSFA.shape[1], MSFA.shape[0]*MSFA.shape[1])).astype(mosaic.dtype)
            for i in range(MSFA.shape[0]):
                for j in range(MSFA.shape[1]):
                    mosaic_numpy[:, :, i*MSFA.shape[1]+j] = mosaic[i::MSFA.shape[0], j::MSFA.shape[1], 0]
            if not os.path.exists(os.path.join(dir_mat, "mosaic")):
                os.mkdir(os.path.join(dir_mat, "mosaic"))
            scio.savemat(os.path.join(dir_mat, "mosaic", f"{idx}.mat"), {'mosaic': mosaic_numpy})
        if args.pan_save == True:
            if not os.path.exists(os.path.join(dir_mat, "pan")):
                os.mkdir(os.path.join(dir_mat, "pan"))
            scio.savemat(os.path.join(dir_mat, "pan", f"{idx}.mat"), {'pan': pan})
        if args.demosaic_save == True:
            if not os.path.exists(os.path.join(dir_mat, "demosaic")):
                os.mkdir(os.path.join(dir_mat, "demosaic"))
            scio.savemat(os.path.join(dir_mat, "demosaic", f"{idx}.mat"), {'demosaic': demosaic})
        if args.simulate == True and args.gt_save == True:
            gt = gts[cnt]
            if not os.path.exists(os.path.join(dir_mat, "gt")):
                os.mkdir(os.path.join(dir_mat, "gt"))
            scio.savemat(os.path.join(dir_mat, "gt", f"{idx}.mat"), {'gt': gt})
        if not os.path.exists(os.path.join(dir_mat, "fused")):
                os.mkdir(os.path.join(dir_mat, "fused"))
        scio.savemat(os.path.join(dir_mat, "fused", f"{idx}.mat"), {'fused': hrms})
    
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
    parser.add_argument('--noise_level', nargs="+", type=float, default=[0.01, 0.05], help='Noise level when generating simulated data.')
    parser.add_argument('--msfa_size', type=int, default=4, help='Size of MSFA')
    parser.add_argument('--data_path', type=str, default="../DataSet/", help='Path of the dataset.')
    parser.add_argument('--data_id', type=str, default=[], nargs="+",
                        help='Index of which data to be tested. If empty, then all be selected.')
    parser.add_argument('--load_ps_model', type=str, default='', help='The pansharpening model to be loaded.')
    parser.add_argument('--load_demosaic_model', type=str, default='', help='The demosaicing model to be loaded.')
    parser.add_argument('--cpu', action='store_true', default=False, help='Determine whether to cpu.')
    parser.add_argument('--device', type=str, default='0', help='Device to train the model.')

    args = parser.parse_args()
    main(args)