import time
import os
import torch
from torch.utils.data import DataLoader
from GetDataSet import MakeSimulateDatasetforDemosaic
import torch.nn as nn
from model import Mpattern_opt, L1_Charbonnier_mean_loss_for_mosaic, reconstruction_loss
import copy
import argparse
import json
import cv2, math
import numpy
import utils
import random
import tqdm
import quality_index

def mosaic(img, MSFA):
    msfa_kernel = torch.zeros(MSFA.shape[0] * MSFA.shape[1], 1, MSFA.shape[0], MSFA.shape[1]).to(img.device)
    for i in range(MSFA.shape[0]):
        for j in range(MSFA.shape[1]):
            msfa_kernel[int(MSFA[i, j]), 0, i, j] += 1
    assert img.shape[1] == msfa_kernel.shape[0] * msfa_kernel.shape[1]
    img_mosaic = torch.nn.functional.conv2d(img, msfa_kernel, bias=None, stride=msfa_kernel.shape[2], groups=img.shape[1])
    img_mosaic = torch.nn.functional.pixel_shuffle(img_mosaic, upscale_factor=msfa_kernel.shape[2])
    return img_mosaic

def get_sparsecube_raw(img_tensor, msfa_size):
    mask = torch.zeros_like(img_tensor)
    for i in range(0, msfa_size):
        for j in range(0, msfa_size):
            mask[:, i * msfa_size + j, i::msfa_size, j::msfa_size] = 1

    return mask.mul(img_tensor), torch.sum(mask.mul(img_tensor), 1).unsqueeze(1)

def input_matrix_wpn(inH, inW, msfa_size):

    h_offset_coord = torch.zeros(inH, inW, 1)
    w_offset_coord = torch.zeros(inH, inW, 1)
    for i in range(0,msfa_size):
        h_offset_coord[i::msfa_size, :, 0] = (i+1)/msfa_size
        w_offset_coord[:, i::msfa_size, 0] = (i+1)/msfa_size

    pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)
    pos_mat = pos_mat.contiguous().view(1, -1, 2)
    return pos_mat


def random_crop_4DTensor(a, crop_size):
    N, C, hei, wid=a.size()
    Height = random.randint(0, hei - crop_size)
    Width = random.randint(0, wid - crop_size)
    return a[:, :, Height:(Height + crop_size), Width:(Width + crop_size)]

def shift_random(x, n_trans=5, max_offset=0):
    H, W = x.shape[-2], x.shape[-1]
    assert n_trans <= H - 1 and n_trans <= W - 1, 'n_shifts should less than {}'.format(H-1)

    if max_offset==0:
        shifts_row = random.sample(list(numpy.concatenate([-1*numpy.arange(1, H), numpy.arange(1, H)])), n_trans)
        shifts_col = random.sample(list(numpy.concatenate([-1*numpy.arange(1, W), numpy.arange(1, W)])), n_trans)
    else:
        assert max_offset<=min(H,W), 'max_offset must be less than min(H,W)'
        shifts_row = random.sample(list(numpy.concatenate([-1*numpy.arange(1, max_offset), numpy.arange(1, max_offset)])), n_trans)
        shifts_col = random.sample(list(numpy.concatenate([-1*numpy.arange(1, max_offset), numpy.arange(1, max_offset)])), n_trans)

    x = torch.cat([x if n_trans == 0 else torch.roll(x, shifts=[sx, sy], dims=[-2, -1]).type_as(x) for sx, sy in
                   zip(shifts_row, shifts_col)], dim=0)
    return x

def transform_opt(HR_4x, train_ps, msfa_size, tran_type):
    if tran_type == 'random':
        tran_type = random.choice([0, 1, 3, 5])
    elif tran_type == 'rotation':
        tran_type = 0
    elif tran_type == 'flip':
        tran_type = 1
    elif tran_type == 'resize':
        tran_type = 3
    elif tran_type == 'globalshift':
        tran_type = 4
    elif tran_type == 'patternshift':
        tran_type = 5
    else:
        raise Exception("wrong tran type")

    if tran_type == 0:
        ## rotation transf
        rn = random.randint(1, 3)
        new_lable = torch.rot90(HR_4x, rn, [2, 3])

    if tran_type == 1:
        if numpy.random.uniform() < 0.5:
            new_lable = torch.flip(HR_4x, [2])
        else:
            new_lable = torch.flip(HR_4x, [3])

    scale_lib = [0.2, 0.25, 0.5, 2, 3, 4]

    if tran_type == 3:
        ## resize transf
        while 1:
            scale_num = random.randint(0, len(scale_lib)-1)
            new_lable = torch.nn.functional.interpolate(HR_4x, scale_factor=scale_lib[scale_num], mode='nearest')
            N, C, H, W = new_lable.size()
            if H >= (train_ps-msfa_size)*scale_lib[0] and H <= train_ps*scale_lib[-1]:  # opt.train_ps*scale_lib[0]*0.5: # value of ICVL_LSA_5_EItrain_Transrandomnew_upd7_alpha1_st1_230721_150236 maybe 40 or opt.train_ps*scale_lib[0]
                break
        new_lable = random_crop_4DTensor(new_lable, (H // msfa_size) * msfa_size)

    if tran_type == 4:
        ## golbal shift transf used in EI
        new_lable = shift_random(HR_4x, n_trans=1, max_offset=0)

    if tran_type == 5:
        ## new shift have just (msfa_size-1)*(msfa_size-1) transformations
        while 1:
            i = random.randint(0, msfa_size-1)
            j = random.randint(0, msfa_size-1)
            if i != 0 or j != 0:
                break
        # print('newshift', i, j)
        new_lable = torch.roll(HR_4x, (-i, -j), (2, 3))
        N, C, H, W = new_lable.size()
        new_lable = new_lable[:, :, 0: (H - msfa_size), 0: (W - msfa_size)]

    return new_lable

def train(demosaic_net: nn.Module, optimizer, train_dataloader, val_dataloader, args):
    print('===>Begin Training!')
    start_epoch = 0
    if args.resume != "":
        start_epoch = int(args.resume) if "best" not in args.resume else int(args.resume.split("_")[-1])

    best_psnr, best_epoch = 0, 0
    MSFA = numpy.array([[0, 1, 2, 3],
                        [4, 5, 6, 7],
                        [8, 9, 10, 11],
                        [12, 13, 14, 15]])
    charbonnier_loss = L1_Charbonnier_mean_loss_for_mosaic(MSFA, next(demosaic_net.parameters()).device)
    reconstruct_loss = reconstruction_loss(MSFA.shape[0])
    t = time.time()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        # train
        demosaic_net.train()
        start_time = time.time()

        # for cnt, data in enumerate(tqdm.tqdm(train_dataloader)):
        for cnt, data in enumerate(train_dataloader):
            lrms_mosaic = data.cuda()

            optimizer.zero_grad()
            scale_coord_map = input_matrix_wpn(lrms_mosaic.shape[2], lrms_mosaic.shape[3], MSFA.shape[0]).to(lrms_mosaic.device)
            lrms_mosaic_up = torch.zeros(lrms_mosaic.shape[0], MSFA.shape[0]*MSFA.shape[1], lrms_mosaic.shape[2], lrms_mosaic.shape[3]).to(lrms_mosaic.device)
            for i in range(MSFA.shape[0]):
                for j in range(MSFA.shape[1]):
                    lrms_mosaic_up[:, i*MSFA.shape[1]+j, i::MSFA.shape[0], j::MSFA.shape[1]] = copy.deepcopy(lrms_mosaic[:, 0, i::MSFA.shape[0], j::MSFA.shape[1]].detach())
            demosaic_first = demosaic_net([lrms_mosaic_up, lrms_mosaic], scale_coord_map)
            loss_raw = charbonnier_loss(demosaic_first, lrms_mosaic)
            loss = loss_raw.clone()

            new_label = transform_opt(demosaic_first.detach(), train_ps=args.train_size, msfa_size=MSFA.shape[0], tran_type="random")
            new_lrms_mosaic_up, new_lrms_mosaic = get_sparsecube_raw(new_label, MSFA.shape[0])
            scale_coord_map = input_matrix_wpn(new_lrms_mosaic.shape[2], new_lrms_mosaic.shape[3], MSFA.shape[0]).to(new_lrms_mosaic.device)
            demosaic_second = demosaic_net([new_lrms_mosaic_up, new_lrms_mosaic], scale_coord_map)
            loss += reconstruct_loss(demosaic_second, new_label)
            loss.backward()
            optimizer.step()

        # val
        psnr_avg = 0.
        demosaic_net.eval()
        with torch.no_grad():
            for cnt, data in enumerate(val_dataloader):
                mosaic, target = data[0].cuda(), data[1].cuda()
                scale_coord_map = input_matrix_wpn(mosaic.shape[2], mosaic.shape[3], MSFA.shape[0]).to(mosaic.device)
                mosaic_up = torch.zeros(mosaic.shape[0], MSFA.shape[0]*MSFA.shape[1], mosaic.shape[2], mosaic.shape[3]).to(mosaic.device)
                for i in range(MSFA.shape[0]):
                    for j in range(MSFA.shape[1]):
                        mosaic_up[:, i*MSFA.shape[1]+j, i::MSFA.shape[0], j::MSFA.shape[1]] = mosaic[:, 0, i::MSFA.shape[0], j::MSFA.shape[1]]
                demosaic = demosaic_net([mosaic_up, mosaic], scale_coord_map).detach()
                psnr_avg += quality_index.calc_psnr(target, demosaic).item()
                
        psnr_avg /= cnt+1

        # save model with highest PSNR
        if psnr_avg > best_psnr:
            best_psnr = psnr_avg
            if best_epoch != 0:
                os.remove(os.path.join(args.dir_model, "best_{}.pth".format(best_epoch)))
            best_epoch = epoch
            torch.save(demosaic_net.state_dict(), os.path.join(args.dir_model, "best_{}.pth".format(epoch)))

        if args.record is not False:
            record = []
            if os.path.exists(args.record):
                with open(args.record, "r") as f:
                    record = json.load(f)
            record.append({"epoch": epoch, "psnr": psnr_avg, "best_psnr": best_psnr, "best_epoch": best_epoch, "learning rate": optimizer.param_groups[0]["lr"]})
            with open(args.record, "w") as f:
                record = json.dump(record, f, indent=2)

        # save model at some frequency
        if epoch % args.save_freq == 0:
            torch.save(demosaic_net.state_dict(), os.path.join(args.dir_model, f"{epoch}.pth"))
        
        # log
        print("Epoch: ", epoch,
            "PSNR: %.4f"%psnr_avg,
            "time: %.2f"%((time.time() - start_time) / 60), "min",
            "best_PSNR: %.4f"%best_psnr,
            "best_epoch: ", best_epoch,
            "learning rate: ",  optimizer.param_groups[0]["lr"]
            )
        
    print(f"Total time: {(time.time() - t) / 60} min")
    print("Best_epoch: {}, PSNR: {}".format(best_epoch, best_psnr))


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

    train_set = MakeSimulateDatasetforDemosaic(args, "train")
    train_dataloader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
    val_set = MakeSimulateDatasetforDemosaic(args, "test")
    val_dataloader = DataLoader(dataset=val_set, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                pin_memory=True)

    total_iterations = args.epochs * len(train_dataloader)
    print('total_iterations:{}'.format(total_iterations))

    demosaic_net = Mpattern_opt(args)

    if args.resume != "":
        backup_pth = os.path.join(dir_model, args.resume + ".pth")
        print("==> Load checkpoint: {}".format(backup_pth))
        demosaic_net.load_state_dict(torch.load(backup_pth), strict=False)
    else:
        print('==> Train from scratch')

    demosaic_net = demosaic_net.cuda()

    optimizer = torch.optim.Adam(demosaic_net.parameters(), args.lr)

    train(demosaic_net, optimizer, train_dataloader, val_dataloader, args)

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
    parser.add_argument('--data_path', type=str, default="../../DataSet/", help='Path of the dataset.')
    parser.add_argument('--record', type=bool, default=True, help='Whether to record the PSNR of each epoch.')

    args = parser.parse_args()
    main(args)