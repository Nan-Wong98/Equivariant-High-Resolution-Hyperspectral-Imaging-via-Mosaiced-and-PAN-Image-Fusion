import os
import numpy
import scipy.io as scio
import cv2
import argparse
import tqdm
import utils
from tqdm.contrib import tzip


def visual_rgb(args):
    if args.data_id == []:
        files = [file for file in os.listdir(args.mat_path) if file.split(".")[-1] == "mat"]
    else:
        files = [file for file in os.listdir(args.mat_path) if
                file in args.data_id and file.split(".")[-1] == "mat"]
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    print(f"Start to visualize RGB: {files}")
    rgb_bands = [0, 1, 13]
    for file in tqdm.tqdm(files):
        id = file.split(".")[0]
        file = os.path.join(args.mat_path, file)
        if args.data_type == "fused":
            mat = scio.loadmat(file)["fused"][:, :, rgb_bands]
            name = f"fused_{id}.png"
        elif args.data_type == "mosaic":
            try:
                mat = scio.loadmat(file)["mosaic"][:, :, rgb_bands]
            except:
                mat = scio.loadmat(file)["gt"][:, :, rgb_bands]
            name = f"mosaic_{id}.png"
        elif args.data_type == "raw_mosaic":
            mat = scio.loadmat(file)["mosaic"]
            mat = utils.pixel_shuffle(mat.transpose(2, 0, 1)[numpy.newaxis,:,:,:], 4)[0].transpose(1, 2, 0)
            name = f"rawmosaic_{id}.png"
        elif args.data_type == "upmosaic":
            mat = scio.loadmat(file)["mosaic"][:, :, rgb_bands]
            dtype = mat.dtype
            mat = cv2.resize(mat, fx=args.spatial_ratio, fy=args.spatial_ratio, dsize=None, interpolation=cv2.INTER_CUBIC)
            if dtype != mat.dtype:
                mat = mat.astype(dtype)
            name = f"upmosaic_{id}.png"
        elif args.data_type == "pan":
            mat = scio.loadmat(file)["pan"]
            name = f"pan_{id}.png"
        else:
            raise ValueError(f"Incorrect data_type: {args.data_type}.")

        if mat.dtype == numpy.int16:
            rgb = numpy.clip(numpy.round(mat / 2**12 * 2 * 255), 0, 255)
        else:
            rgb = numpy.clip(numpy.round(mat*255), 0, 255)
        cv2.imwrite(os.path.join(args.save_path, name), rgb)

        if rgb.shape[-1] == 1:
            rgb = numpy.concatenate((rgb, rgb, rgb), axis=-1)
        
        if args.detach == True:
            detach1, detach2, data_with_box, args.detach_size, args.detach_coordinate = utils.region_detach(rgb, args.detach_size,
                                                                                                  args.detach_coordinate,
                                                                                                  box_width=args.boxwidth,
                                                                                                  rgb=args.boxcolor)

            detach_name1 = f"detach_{id}_{str(args.detach_size[0])}_{str(args.detach_size[1])}_{str(args.detach_coordinate[0])}_{str(args.detach_coordinate[1])}_{str(args.boxwidth)}.png"
            detach_name2 = f"detach_{id}_{str(args.detach_size[0])}_{str(args.detach_size[1])}_{str(args.detach_coordinate[2])}_{str(args.detach_coordinate[3])}_{str(args.boxwidth)}.png"
            data_with_box_name = name[:-4] + f"_box" + name[-4:]
            cv2.imwrite(os.path.join(args.save_path, detach_name1), detach1)
            cv2.imwrite(os.path.join(args.save_path, detach_name2), detach2)
            cv2.imwrite(os.path.join(args.save_path, data_with_box_name), data_with_box)


def visual_diffmap(args):  
    fused_files = [file for file in os.listdir(args.mat_path) if
                   file in args.data_id and file.split(".")[-1] == "mat"]
    gt_files = [file for file in os.listdir(args.mat_path_for_diff) if
                file in args.data_id and file.split(".")[-1] == "mat"]
    if fused_files == []:
        fused_files = [file for file in os.listdir(args.mat_path) if file.split(".")[-1] == "mat"]
        gt_files = [file for file in os.listdir(args.mat_path_for_diff) if file.split(".")[-1] == "mat"]

    assert fused_files == gt_files, "Mismatch between fused_files and PAN gt_files!"

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    print(f"Start to visualize Diff_map: {[os.path.join(args.mat_path, file) for file in fused_files]}")
    for fused_file, gt_file in tqdm.tqdm(tzip(fused_files, gt_files)):
        id = fused_file.split(".")[0]
        fused_file = os.path.join(args.mat_path, fused_file)
        gt_file = os.path.join(args.mat_path_for_diff, gt_file)

        fused = scio.loadmat(fused_file)["fused"]
        gt = scio.loadmat(gt_file)["gt"].astype(numpy.float32)
        
        # plot mae map
        mae_diff = numpy.mean(numpy.abs(gt - fused), axis=-1, keepdims=True)
        mae_diff = numpy.clip(numpy.round(mae_diff * 255), 0, 255)
        utils.plt_diff(mae_diff, os.path.join(args.save_path, f"mae_{id}.png"), upper_boundary=args.mae_level)
        if args.detach == True:
            rgb = cv2.imread(os.path.join(args.save_path, f"mae_{id}.png"))
            detach1, detach2, data_with_box, args.detach_size, args.detach_coordinate = utils.region_detach(rgb, args.detach_size,
                                                                                                  args.detach_coordinate,
                                                                                                  box_width=args.boxwidth,
                                                                                                  rgb=args.boxcolor)

            detach_name1 = f"mae_detach_{id}_{str(args.detach_size[0])}_{str(args.detach_size[1])}_{str(args.detach_coordinate[0])}_{str(args.detach_coordinate[1])}_{str(args.boxwidth)}.png"
            detach_name2 = f"mae_detach_{id}_{str(args.detach_size[0])}_{str(args.detach_size[1])}_{str(args.detach_coordinate[2])}_{str(args.detach_coordinate[3])}_{str(args.boxwidth)}.png"
            cv2.imwrite(os.path.join(args.save_path, detach_name1), detach1)
            cv2.imwrite(os.path.join(args.save_path, detach_name2), detach2)
            cv2.imwrite(os.path.join(args.save_path, f"mae_{id}_box.png"), data_with_box)

        # plot sam map
        sam_diff = (gt*fused).sum(axis=-1, keepdims=True)/(numpy.sqrt((gt**2).sum(axis=-1, keepdims=True))*\
                   numpy.sqrt((fused**2).sum(axis=-1, keepdims=True)))
        sam_diff = numpy.arccos(sam_diff)
        sam_diff = numpy.clip(numpy.round(sam_diff / numpy.pi * 255), 0, 255)
        utils.plt_diff(sam_diff, os.path.join(args.save_path, f"sam_{id}.png"), upper_boundary=args.sam_level)

        if args.detach == True:
            rgb = cv2.imread(os.path.join(args.save_path, f"sam_{id}.png"))
            detach1, detach2, data_with_box, args.detach_size, args.detach_coordinate = utils.region_detach(rgb, args.detach_size,
                                                                                                  args.detach_coordinate,
                                                                                                  box_width=args.boxwidth,
                                                                                                  rgb=args.boxcolor)

            detach_name1 = f"sam_detach_{id}_{str(args.detach_size[0])}_{str(args.detach_size[1])}_{str(args.detach_coordinate[0])}_{str(args.detach_coordinate[1])}_{str(args.boxwidth)}.png"
            detach_name2 = f"sam_detach_{id}_{str(args.detach_size[0])}_{str(args.detach_size[1])}_{str(args.detach_coordinate[2])}_{str(args.detach_coordinate[3])}_{str(args.boxwidth)}.png"
            cv2.imwrite(os.path.join(args.save_path, detach_name1), detach1)
            cv2.imwrite(os.path.join(args.save_path, detach_name2), detach2)
            cv2.imwrite(os.path.join(args.save_path, f"sam_{id}_box.png"), data_with_box)

        if not os.path.exists(os.path.join(args.save_path, "colorbar.png")):
            utils.plt_colorbar(os.path.join(args.save_path, "colorbar.png"))
        if not os.path.exists(os.path.join(args.save_path, "gt.png")):
            utils.plt_diff(numpy.zeros_like(sam_diff), os.path.join(args.save_path, "gt.png"))
            if args.detach == True:
                rgb = cv2.imread(os.path.join(args.save_path, f"gt.png"))
                rgb = cv2.resize(rgb, dsize=(mae_diff.shape[1], mae_diff.shape[0]), interpolation=cv2.INTER_CUBIC)
                detach1, detach2, data_with_box, args.detach_size, args.detach_coordinate = utils.region_detach(rgb, args.detach_size,
                                                                                                    args.detach_coordinate,
                                                                                                    box_width=args.boxwidth,
                                                                                                    rgb=args.boxcolor)

                detach_name1 = f"gt_detach_{id}_{str(args.detach_size[0])}_{str(args.detach_size[1])}_{str(args.detach_coordinate[0])}_{str(args.detach_coordinate[1])}_{str(args.boxwidth)}.png"
                detach_name2 = f"gt_detach_{id}_{str(args.detach_size[0])}_{str(args.detach_size[1])}_{str(args.detach_coordinate[2])}_{str(args.detach_coordinate[3])}_{str(args.boxwidth)}.png"
                cv2.imwrite(os.path.join(args.save_path, detach_name1), detach1)
                cv2.imwrite(os.path.join(args.save_path, detach_name2), detach2)
                cv2.imwrite(os.path.join(args.save_path, f"gt_box.png"), data_with_box)



def main(args):
    if args.visual_task == "rgb":
        visual_rgb(args)
    elif args.visual_task == "diffmap":
        visual_diffmap(args)
    else:
        raise ValueError(f"Incorrect `visual_task`: {args.visual_task}. Set to `rgb`, or `diffmap`  instead")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--visual_task', type=str, default="diffmap", help='Determine to visualize rgb or diff_map.')
    parser.add_argument('--spatial_ratio', type=int, default=8, help='Ratio of spatial resolutions between MS and PAN')
    parser.add_argument('--mat_path', type=str, default="./WV2/1/simulate/mat/", help='Path to load mat files.')
    parser.add_argument('--mat_path_for_diff', type=str, default="../DataSet/WV2/test/ms_256/",
                        help='Path to load mat files (as the target). Only effective when `visual_task` is True')
    parser.add_argument('--save_path', type=str, default="./WV2/1/simulate/rgb/", help='Path to save the rgb files.')
    parser.add_argument('--data_id', type=str, default=[], nargs="+",
                        help='Index of which data to be visualized. If empty, then all be selected.')
    parser.add_argument('--data_type', type=str, default="fused",
                        help='(1)ms (2)upms (3)lrms (4)pan (5)lrpan (6)fused, for different data types')
    parser.add_argument('--detach', action='store_true', default=False,
                        help='Determine whether to detach a local region. Effective only when `visual_task` is `rgb`')
    parser.add_argument('--detach_size', type=int, default=32, nargs="+",
                        help='Size of the local region to be detached.')
    parser.add_argument('--detach_coordinate', type=int, default=0, nargs="+",
                        help='Coordinate of the left-bottom of the zoom-in region.')
    parser.add_argument('--boxwidth', type=int, default=5, help='Width of the detached box.')
    parser.add_argument('--boxcolor', type=str, default="r", help='Color of the first detached box, oppsite to the other one.')
    parser.add_argument('--mae_level', type=int, default=16, help='Control level of the generated mae map.')
    parser.add_argument('--sam_level', type=int, default=16, help='Control level of the generated sam map.')

    args = parser.parse_args()
    main(args)