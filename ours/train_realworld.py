import time
import os
import torch
from torch.utils.data import DataLoader
from GetDataSet import MakeRealWorldDataset
import torch.nn as nn
from model import Network, Degrade_R, degrade_dm, transform
import copy
import argparse
import json
import cv2
import numpy
import utils
import random, tqdm
import quality_index
import math

def train(fuse_net: nn.Module, degrade_r: nn.Module, optimizers, train_dataloader, val_dataloader, args):
    print('===>Begin Training!')
    start_epoch = 0
    if args.resume != "":
        start_epoch = int(args.resume) if "best" not in args.resume else int(args.resume.split("_")[-1])

    t = time.time()
    device = next(fuse_net.parameters()).device
    MSFA = numpy.array([[0, 1, 2, 3],
                        [4, 5, 6, 7],
                        [8, 9, 10, 11],
                        [12, 13, 14, 15]])
    msfa_kernel = torch.zeros(MSFA.shape[0] * MSFA.shape[1], 1, MSFA.shape[0]*2, MSFA.shape[1]*2).to(device)
    for i in range(MSFA.shape[0]):
        for j in range(MSFA.shape[1]):
            msfa_kernel[int(MSFA[i, j]), 0, i*2, j*2] = 0.25
            msfa_kernel[int(MSFA[i, j]), 0, i*2+1, j*2] = 0.25
            msfa_kernel[int(MSFA[i, j]), 0, i*2, j*2+1] = 0.25
            msfa_kernel[int(MSFA[i, j]), 0, i*2+1, j*2+1] = 0.25
    best_qnr = 0
    best_epoch = 0
    numpy.set_printoptions(precision=3, suppress=True)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        # train
        fuse_net.train()
        degrade_r.train()
        start_time = time.time()

        loss_per_epoch = 0
        for cnt, data in enumerate(tqdm.tqdm(train_dataloader)):
        # for cnt, data in enumerate(train_dataloader):
            mosaic, pan = data[0].to(device), data[1].to(device)

            hrms_first = fuse_net(mosaic, pan)
            pan_from_hrms = degrade_r(hrms_first)
            mosaic_from_hrms = degrade_dm(hrms_first, msfa_kernel)
            loss_mosaic = nn.functional.mse_loss(mosaic_from_hrms, mosaic)
            loss_pan = nn.functional.mse_loss(pan_from_hrms, pan)

            hrms_first = transform(hrms_first, msfa_size=4, spatial_ratio=2).detach()
            # optimize demosaic network
            pan_from_hrms = degrade_r(hrms_first)
            mosaic_from_hrms = degrade_dm(hrms_first, msfa_kernel)
            hrms_second = fuse_net(mosaic_from_hrms, pan_from_hrms)

            optimizers[0].zero_grad()
            optimizers[1].zero_grad()
            loss_ei = torch.nn.functional.mse_loss(hrms_second, hrms_first)
            loss = loss_ei + loss_mosaic + loss_pan
            loss.backward()
            optimizers[0].step()
            optimizers[1].step()

            loss_per_epoch += loss.detach().item()

        loss_per_epoch /= cnt + 1
        qnr_avg, d_lambda_avg, d_s_avg = 0., 0., 0.
        # val
        fuse_net.eval()
        with torch.no_grad():
            for cnt, data in enumerate(val_dataloader):
                mosaic, pan = data[0].to(device), data[1].to(device)
                fused = fuse_net(mosaic, pan).detach()

                mosaic_pred = degrade_dm(fused, msfa_kernel).detach()
                pan_pred = degrade_r(fused).detach()
                fused = torch.clamp(torch.round(fused*2**12), 0, 2**12).short()
                
                qnr, d_lambda, d_s = quality_index.calc_qnr_mosaic(fused.double()/2**12, mosaic.double(), pan.double(), msfa_kernel, patch_size=32, scale_factor=args.spatial_ratio)
                
                qnr_avg += qnr.item()
                d_lambda_avg += d_lambda.item()
                d_s_avg += d_s.item()

                if args.visual == True and epoch % args.visual_freq == 0 and epoch != 0:
                    rgb_bands = [0, 1, 13]
                    if not os.path.exists(os.path.join(args.dir_record, str(cnt))):
                        os.mkdir(os.path.join(args.dir_record, str(cnt)))

                    if not os.path.exists(os.path.join(args.dir_record, str(cnt), "mosaic.png")):
                        mosaic_numpy = utils.pixel_shuffle_inv(mosaic.cpu().numpy(), 4)[0].transpose(1, 2, 0)[:, :, rgb_bands]
                        cv2.imwrite(os.path.join(args.dir_record, str(cnt), "mosaic.png"), numpy.clip(numpy.round(mosaic_numpy * 255), 0, 255))
                        pan_numpy = pan.cpu().numpy()[0].transpose(1, 2, 0)
                        cv2.imwrite(os.path.join(args.dir_record, str(cnt), "pan.png"), numpy.clip(numpy.round(pan_numpy * 255), 0, 255))    
                    # mosaic_pred_numpy = mosaic_pred.cpu().numpy()[0].transpose(1, 2, 0)
                    mosaic_pred_numpy = utils.pixel_shuffle_inv(mosaic_pred.cpu().numpy(), 4)[0].transpose(1, 2, 0)
                    cv2.imwrite(os.path.join(args.dir_record, str(cnt), f"mosaic_pred_{epoch}.png"), numpy.clip(numpy.round(mosaic_pred_numpy[:, :, rgb_bands] * 255), 0, 255))
                    pan_pred_numpy = pan_pred.cpu().numpy()[0].transpose(1, 2, 0)
                    cv2.imwrite(os.path.join(args.dir_record, str(cnt), f"pan_pred_{epoch}.png"), numpy.clip(numpy.round(pan_pred_numpy * 255), 0, 255))
                    fused_numpy = fused[0].permute(1, 2, 0).cpu().numpy()[:, :, rgb_bands]
                    cv2.imwrite(os.path.join(args.dir_record, str(cnt), f"fused_{epoch}.png"), numpy.clip(numpy.round(fused_numpy / 2**12 * 255), 0, 255))

        qnr_avg /= cnt+1
        d_lambda_avg /= cnt+1
        d_s_avg /= cnt+1

        # save model with highest QNR
        if qnr_avg > best_qnr:
            best_qnr = qnr_avg
            if best_epoch != 0:
                os.remove(os.path.join(args.dir_model, "best_{}.pth".format(best_epoch)))
            best_epoch = epoch
            torch.save({"net": fuse_net.state_dict(), "degrade_r": degrade_r.state_dict()}, os.path.join(args.dir_model, "best_{}.pth".format(epoch)))
        
        if args.record is not False:
            record = []
            if os.path.exists(args.record):
                with open(args.record, "r") as f:
                    record = json.load(f)
            record.append({"epoch": epoch,
                           "loss": loss_per_epoch,
                           "qnr": qnr_avg,
                           "best_qnr": best_qnr,
                           "best_epoch": best_epoch,
                           "learning rate": optimizers[0].param_groups[0]["lr"],
                           "R_estimate": degrade_r.spec_res.weight.detach().flatten().cpu().numpy().tolist(),
                           })
            with open(args.record, "w") as f:
                record = json.dump(record, f, indent=2)
        
        # save model at some frequency
        if epoch % args.save_freq == 0:
            torch.save({"net": fuse_net.state_dict(), "degrade_r": degrade_r.state_dict()}, os.path.join(args.dir_model, f"{epoch}.pth"))

        # log
        print("Epoch: ", epoch,
            "loss: %.4f"%loss_per_epoch,
            "time: %.2f"%((time.time() - start_time) / 60), "min",
            "loss: %.4f"%loss_per_epoch,
            "qnr: %.4f"%qnr_avg,
            "best_qnr: %.4f"%best_qnr,
            "best_epoch: ", best_epoch,
            "learning rate: ", optimizers[0].param_groups[0]["lr"], "\n",
            )
        loss_per_epoch = 0 

    print(f"Total time: {(time.time() - t) / 60} min")
    print("Best epoch: {}, Best QNR: {}".format(best_epoch, best_qnr))

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

    total_iterations = args.epochs * args.iters_per_epoch
    print('total_iterations:{}'.format(total_iterations))

    train_set = MakeRealWorldDataset(args, "train")
    train_dataloader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
    val_set = MakeRealWorldDataset(args, "test")
    val_dataloader = DataLoader(dataset=val_set, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                pin_memory=True)

    fuse_net = Network(args)
    degrade_r = Degrade_R(args)

    if args.resume != "":
        backup_pth = os.path.join(dir_model, args.resume + ".pth")
        print("==> Load checkpoint: {}".format(backup_pth))
        fuse_net.load_state_dict(torch.load(backup_pth)["net"], strict=False)
        degrade_r.load_state_dict(torch.load(backup_pth)["degrade_r"], strict=False)
    else:
        print('==> Train from scratch')
    
    fuse_net = fuse_net.to(f"cuda:{args.device}")
    degrade_r = degrade_r.to(f"cuda:{args.device}")

    optimizer1 = torch.optim.Adam(fuse_net.parameters(), args.lr)
    optimizer2 = torch.optim.Adam(degrade_r.parameters(), args.lr)

    train(fuse_net, degrade_r, [optimizer1, optimizer2], train_dataloader, val_dataloader, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--idx', type=int, default=1, help='Index to identify models.')
    parser.add_argument('--dataset', type=str, default="Ours", help='Dataset to be loaded.')
    parser.add_argument('--train_size', type=int, default=128, help='Size of the training image in a batch.')
    parser.add_argument('--msfa_size', type=int, default=4, help='Size of MSFA')
    parser.add_argument('--spatial_ratio', type=int, default=2, help='Ratio of spatial resolutions between MS and PAN')
    parser.add_argument('--num_bands', type=int, default=16, help='Number of bands of a MS image.')
    parser.add_argument('--stride', type=int, default=64, help='Stride when crop an original image into patches.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training dataset.')
    parser.add_argument('--epochs', type=int, default=1000, help='Total epochs to train the model.')
    parser.add_argument('--iters_per_epoch', type=int, default=100, help='Iteration steps per epoch.')
    parser.add_argument('--save_freq', type=int, default=50, help='Save the checkpoints of the model every [save_freq] epochs.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer.')
    parser.add_argument('--lr_decay', action="store_true", help='Determine if to decay the learning rate.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate to train the model.')
    parser.add_argument('--device', type=str, default='0', help='Device to train the model.')
    parser.add_argument('--num_workers', type=int, default=1, help='Num_workers to train the model.')
    parser.add_argument('--resume', type=str, default='', help='Index of the model to be resumed, eg. 1000.')
    parser.add_argument('--resume_demosaic', type=str, default='', help='Index of the demosaicing model to be resumed, eg. 1000.')
    parser.add_argument('--data_path', type=str, default="../DataSet/", help='Path of the dataset.')
    parser.add_argument('--record', type=bool, default=True, help='Whether to record the PSNR of each epoch.')
    parser.add_argument('--visual', action='store_true', default=False, help='Whether to visualize the validation data.')
    parser.add_argument('--visual_freq', type=int, default=50, help='Frequency to visualize the validation data.')

    args = parser.parse_args()
    main(args)