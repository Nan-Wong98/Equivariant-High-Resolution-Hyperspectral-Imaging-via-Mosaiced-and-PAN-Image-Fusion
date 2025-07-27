import time
import os
import torch
from torch.utils.data import DataLoader
from GetDataSet import MakeRealWorldDatasetForPansharpening
import torch.nn as nn
from model import SFNet, Generator, Discriminator_spe, Discriminator_spa, hp
import copy
import argparse
import json
import cv2
import numpy
import utils
import random
import tqdm
from pytorch_msssim import ssim
import quality_index

def train(ps_net: nn.Module, dis_spe: nn.Module, dis_spa: nn.Module, optimizers, train_dataloader, val_dataloader, args):
    print('===>Begin Training!')
    start_epoch = 0
    if args.resume != "":
        start_epoch = int(args.resume) if "best" not in args.resume else int(args.resume.split("_")[-1])

    best_epoch, best_qnr = 0, 0
    MSFA = numpy.array([[0, 1, 2, 3],
                        [4, 5, 6, 7],
                        [8, 9, 10, 11],
                        [12, 13, 14, 15]])
    device = next(ps_net.parameters()).device
    msfa_kernel = torch.zeros(MSFA.shape[0] * MSFA.shape[1], 1, MSFA.shape[0]*2, MSFA.shape[1]*2).to(device)
    for i in range(MSFA.shape[0]):
        for j in range(MSFA.shape[1]):
            msfa_kernel[int(MSFA[i, j]), 0, i*2, j*2] = 0.25
            msfa_kernel[int(MSFA[i, j]), 0, i*2+1, j*2] = 0.25
            msfa_kernel[int(MSFA[i, j]), 0, i*2, j*2+1] = 0.25
            msfa_kernel[int(MSFA[i, j]), 0, i*2+1, j*2+1] = 0.25
    t = time.time()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        # train
        ps_net.train()
        start_time = time.time()

        for cnt, data in enumerate(tqdm.tqdm(train_dataloader)):
            lrms, pan = data[0].to(device), data[1].to(device)

            optimizers[1].zero_grad()
            upms = torch.nn.functional.interpolate(lrms, scale_factor=pan.shape[-1]//lrms.shape[-1], mode="bilinear")
            output_from_dis_spe = dis_spe(upms)
            b = (torch.ones(output_from_dis_spe.shape) * (.7 + torch.rand(1) * .5)).to(output_from_dis_spe.device)
            loss = torch.mean(torch.sum((output_from_dis_spe-b)**2, (1,2,3)))
            loss.backward()
            optimizers[1].step()

            optimizers[2].zero_grad()
            output_from_dis_spa = dis_spa(pan)
            b = (torch.ones(output_from_dis_spa.shape) * (.7 + torch.rand(1) * .5)).to(output_from_dis_spa.device)
            loss = torch.mean(torch.sum((output_from_dis_spa-b)**2, (1,2,3)))
            loss.backward()
            optimizers[2].step()

            optimizers[0].zero_grad()
            fused = ps_net(lrms, pan)
            lrms_from_fused = torch.nn.functional.interpolate(fused, scale_factor=lrms.shape[-1]/fused.shape[-1], mode="bilinear")
            loss_spe = torch.mean(torch.sum((lrms_from_fused-lrms)**2, (1,2,3)))
            output_from_dis_spe = dis_spe(fused)
            c = (torch.ones(output_from_dis_spe.shape) * (.7 + torch.rand(1) * .5)).to(output_from_dis_spe.device)
            loss_adv1 = torch.mean(torch.sum((output_from_dis_spe-c)**2, (1,2,3)))

            pan_from_fused = torch.mean(fused, 1, keepdim=True)
            hp_from_pan = hp(pan)
            hp_from_pan_from_fused = hp(pan_from_fused)
            loss_spa = torch.mean(torch.sum((hp_from_pan_from_fused-hp_from_pan)**2, (1,2,3)))
            output_from_dis_spa = dis_spa(pan_from_fused)
            d = (torch.ones(output_from_dis_spa.shape) * (.7 + torch.rand(1) * .5)).to(output_from_dis_spe.device)
            loss_adv2 = torch.mean(torch.sum((output_from_dis_spa-d)**2, (1,2,3)))

            loss = loss_spe + 0.002*loss_adv1 + 5*loss_spa + 0.001*loss_adv2
            loss.backward()
            optimizers[0].step()

            fused = ps_net(lrms, pan).detach()

            optimizers[1].zero_grad()
            output_from_dis_spe = dis_spe(fused)
            a = (torch.ones(output_from_dis_spa.shape) * (torch.rand(1) * .3)).to(output_from_dis_spe.device)
            loss = torch.mean(torch.sum((output_from_dis_spe-a)**2, (1,2,3)))
            loss.backward()
            optimizers[1].step()

            optimizers[2].zero_grad()
            pan_from_fused = torch.mean(fused, 1, keepdim=True)
            output_from_dis_spa = dis_spa(pan_from_fused)
            a = (torch.ones(output_from_dis_spa.shape) * (torch.rand(1) * .3)).to(output_from_dis_spa.device)
            loss = torch.mean(torch.sum((output_from_dis_spa-a)**2, (1,2,3)))
            loss.backward()
            optimizers[2].step()

        # val
        qnr_avg, d_lambda_avg, d_s_avg = 0., 0., 0.
        ps_net.eval()
        with torch.no_grad():
            for cnt, data in enumerate(val_dataloader):
                mosaic, lrms, pan = data[0].to(device), data[1].to(device), data[2].to(device)

                fused = ps_net(lrms, pan).detach()

                fused = torch.clamp(torch.round(fused*2**12), 0, 2**12).short()

                qnr, d_lambda, d_s = quality_index.calc_qnr_mosaic(fused.double()/2**12, mosaic.double(), pan.double(), msfa_kernel, patch_size=32, scale_factor=args.spatial_ratio)
                qnr_avg += qnr.item()
                d_lambda_avg += d_lambda.item()
                d_s_avg += d_s.item()

                if args.visual == True and epoch % args.visual_freq == 0 and epoch != 0:
                    rgb_bands = [0, 1, 13]
                    if not os.path.exists(os.path.join(args.dir_record, str(cnt))):
                        os.mkdir(os.path.join(args.dir_record, str(cnt)))
                    if not os.path.exists(os.path.join(args.dir_record, str(cnt), "mosaic.png")) or epoch == args.visual_freq:
                        mosaic_numpy = utils.pixel_shuffle_inv(mosaic.cpu().numpy(), 4)[0].transpose(1, 2, 0)[:, :, rgb_bands]
                        mosaic_numpy = numpy.clip(numpy.round(mosaic_numpy * 255), 0, 255)
                        cv2.imwrite(os.path.join(args.dir_record, str(cnt), "mosaic.png"), mosaic_numpy)
                        lrms_numpy = lrms[0].permute(1, 2, 0).cpu().numpy()[:, :, rgb_bands]
                        lrms_numpy = numpy.clip(numpy.round(lrms_numpy * 255), 0, 255)
                        cv2.imwrite(os.path.join(args.dir_record, str(cnt), "demosaic.png"), lrms_numpy)
                        pan_numpy = pan.cpu().numpy()[0].transpose(1, 2, 0)
                        pan_numpy = numpy.clip(numpy.round(pan_numpy * 255), 0, 255)
                        cv2.imwrite(os.path.join(args.dir_record, str(cnt), "pan.png"), pan_numpy)
                    
                    fused_numpy = fused[0].permute(1, 2, 0).cpu().numpy()[:, :, rgb_bands]
                    fused_numpy = numpy.clip(numpy.round(fused_numpy / 2**12 * 255), 0, 255)
                    cv2.imwrite(os.path.join(args.dir_record, str(cnt), f"fused_{epoch}.png"), fused_numpy)

        qnr_avg /= cnt+1
        d_lambda_avg /= cnt+1
        d_s_avg /= cnt+1

        # save model with highest QNR
        if qnr_avg > best_qnr:
            best_qnr = qnr_avg
            if best_epoch != 0:
                os.remove(os.path.join(args.dir_model, "best_{}.pth".format(best_epoch)))
            best_epoch = epoch
            torch.save({"psnet": ps_net.state_dict(),
                        "dis_spe": dis_spe.state_dict(),
                        "dis_spa": dis_spa.state_dict()}, os.path.join(args.dir_model, "best_{}.pth".format(epoch)))

        if args.record is not False:
            record = []
            if os.path.exists(args.record):
                with open(args.record, "r") as f:
                    record = json.load(f)
            record.append({"epoch": epoch, "qnr": qnr_avg, "best_qnr": best_qnr, "best_epoch": best_epoch, "learning rate": optimizers[0].param_groups[0]["lr"]})
            with open(args.record, "w") as f:
                record = json.dump(record, f, indent=2)

        # save model at some frequency
        if epoch % args.save_freq == 0:
            torch.save({"psnet": ps_net.state_dict(),
                        "dis_spe": dis_spe.state_dict(),
                        "dis_spa": dis_spa.state_dict()}, os.path.join(args.dir_model, f"{epoch}.pth"))
        
        # log
        print("Epoch: ", epoch,
            "QNR: %.4f"%qnr_avg,
            "time: %.2f"%((time.time() - start_time) / 60), "min", 
            "best_QNR: %.4f"%best_qnr,
            "best_epoch: ", best_epoch,
            "learning rate: ", optimizers[0].param_groups[0]["lr"]
            )

    print(f"Total time: {(time.time() - t) / 60} min")
    print("Best_epoch: {}, QNR: {}".format(best_epoch, best_qnr))

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

    demosaic_net = SFNet(args)
    demosaic_net.load_state_dict(torch.load(args.resume_demosaic), strict=False)
    demosaic_net = demosaic_net.to(f"cuda:{args.device}")
    train_set = MakeRealWorldDatasetForPansharpening(args, "train", demosaic_net)
    train_dataloader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
    val_set = MakeRealWorldDatasetForPansharpening(args, "test", demosaic_net)
    val_dataloader = DataLoader(dataset=val_set, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                pin_memory=True)

    ps_net = Generator(args)
    dis_spe, dis_spa = Discriminator_spe(args), Discriminator_spa(args)

    if args.resume != "":
        backup_pth = os.path.join(dir_model, args.resume + ".pth")
        print("==> Load checkpoint: {}".format(backup_pth))
        ps_net.load_state_dict(torch.load(backup_pth), strict=False)
    else:
        print('==> Train from scratch')

    ps_net = ps_net.to(f"cuda:{args.device}")
    dis_spe, dis_spa = dis_spe.to(f"cuda:{args.device}"), dis_spa.to(f"cuda:{args.device}")

    optimizer0 = torch.optim.Adam(ps_net.parameters(), args.lr)
    optimizer1 = torch.optim.Adam(dis_spe.parameters(), args.lr)
    optimizer2 = torch.optim.Adam(dis_spa.parameters(), args.lr)

    train(ps_net, dis_spe, dis_spa, [optimizer0, optimizer1, optimizer2], train_dataloader, val_dataloader, args)

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
    parser.add_argument('--epochs', type=int, default=1000, help='Total epochs to train the model.')
    parser.add_argument('--iters_per_epoch', type=int, default=100, help='Iteration steps per epoch.')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='Save the checkpoints of the model every [save_freq] epochs.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate to train the model.')
    parser.add_argument('--lr_decay', action="store_true", help='Determine if to decay the learning rate.')
    parser.add_argument('--device', type=str, default='0', help='Device to train the model.')
    parser.add_argument('--num_workers', type=int, default=1, help='Num_workers to train the model.')
    parser.add_argument('--resume', type=str, default='', help='Index of the model to be resumed, eg. 1000.')
    parser.add_argument('--resume_demosaic', type=str, default='', help='Index of the demosaicing model to be resumed, eg. 1000.')
    parser.add_argument('--data_path', type=str, default="./", help='Path of the dataset.')
    parser.add_argument('--record', type=bool, default=True, help='Whether to record the PSNR of each epoch.')
    parser.add_argument('--visual', action='store_true', default=False, help='Whether to visualize the validation data.')
    parser.add_argument('--visual_freq', type=int, default=50, help='Frequency to visualize the validation data.')

    args = parser.parse_args()
    main(args)