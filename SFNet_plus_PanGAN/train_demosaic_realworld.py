import time
import os
import torch
from torch.utils.data import DataLoader
from GetDataSet import MakeRealWorldDatasetforDemosaic
import torch.nn as nn
from model import SFNet
import copy
import argparse
import json
import cv2, math
import numpy
import utils
import random
import tqdm
import quality_index

def train(demosaic_net: nn.Module, optimizer, train_dataloader, val_dataloader, args):
    print('===>Begin Training!')
    start_epoch = 0
    if args.resume != "":
        start_epoch = int(args.resume) if "best" not in args.resume else int(args.resume.split("_")[-1])

    best_psnr, best_epoch = -10, 0
    MSFA = numpy.array([[0, 1, 2, 3],
                        [4, 5, 6, 7],
                        [8, 9, 10, 11],
                        [12, 13, 14, 15]])
    mse_loss = nn.MSELoss()
    t = time.time()
    device = next(demosaic_net.parameters()).device
    for epoch in range(start_epoch + 1, args.epochs + 1):
        # train
        demosaic_net.train()
        start_time = time.time()

        for cnt, data in enumerate(tqdm.tqdm(train_dataloader)):
            mosaic, gt = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            demosaic = demosaic_net(mosaic)

            loss = mse_loss(demosaic, gt)
            loss.backward()
            optimizer.step()

        # val
        psnr_avg = 0.
        demosaic_net.eval()
        with torch.no_grad():
            for cnt, data in enumerate(val_dataloader):
                mosaic, gt = data[0].to(device), data[1].to(device)
                demosaic = demosaic_net(mosaic).detach()
                psnr_avg += quality_index.calc_psnr(gt, demosaic).item()

                if args.visual == True and epoch % args.visual_freq == 0 and epoch != 0:
                    rgb_bands = [0, 1, 13]
                    if not os.path.exists(os.path.join(args.dir_record, str(cnt))):
                        os.mkdir(os.path.join(args.dir_record, str(cnt)))
                    if not os.path.exists(os.path.join(args.dir_record, str(cnt), "mosaic.png")) or epoch == args.visual_freq:
                        mosaic_numpy = utils.pixel_shuffle_inv(mosaic.cpu().numpy(), 4)[0].transpose(1, 2, 0)[:, :, rgb_bands]
                        mosaic_numpy = numpy.clip(numpy.round(mosaic_numpy * 255), 0, 255)
                        cv2.imwrite(os.path.join(args.dir_record, str(cnt), "mosaic.png"), mosaic_numpy)
                    
                    demosaic_numpy = demosaic[0].permute(1, 2, 0).cpu().numpy()[:, :, rgb_bands]
                    demosaic_numpy = numpy.clip(numpy.round(demosaic_numpy * 255), 0, 255)
                    cv2.imwrite(os.path.join(args.dir_record, str(cnt), f"demosaic_{epoch}.png"), demosaic_numpy)
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

    train_set = MakeRealWorldDatasetforDemosaic(args, "train")
    train_dataloader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
    val_set = MakeRealWorldDatasetforDemosaic(args, "test")
    val_dataloader = DataLoader(dataset=val_set, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                pin_memory=True)

    total_iterations = args.epochs * len(train_dataloader)
    print('total_iterations:{}'.format(total_iterations))

    demosaic_net = SFNet(args)

    if args.resume != "":
        backup_pth = os.path.join(dir_model, args.resume + ".pth")
        print("==> Load checkpoint: {}".format(backup_pth))
        demosaic_net.load_state_dict(torch.load(backup_pth), strict=False)
    else:
        print('==> Train from scratch')

    demosaic_net = demosaic_net.to(f"cuda:{args.device}")

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
    parser.add_argument('--data_path', type=str, default="../DataSet/", help='Path of the dataset.')
    parser.add_argument('--record', type=bool, default=True, help='Whether to record the PSNR of each epoch.')
    parser.add_argument('--visual', action='store_true', default=False, help='Whether to visualize the validation data.')
    parser.add_argument('--visual_freq', type=int, default=50, help='Frequency to visualize the validation data.')

    args = parser.parse_args()
    main(args)