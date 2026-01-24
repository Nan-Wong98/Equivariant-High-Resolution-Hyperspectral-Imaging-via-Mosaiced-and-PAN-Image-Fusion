# 📖 Introduction

Compared to competing methods, this proposed method is a one-step method.

---

## 📂 Dataset Structure

The dataset is organized as follows:
```
Dataset/
├── CAVE/
│ ├── train/
│ └── test/
├── ICVL/
│ ├── train/
│ └── test/
└── real_world/
├── train/
└── test/
```

## 🚀 Usage

The command lines for **training**, **generation**, **testing**, and **visualization** are provided below.


# 1. Train
## 🟢 CAVE 
```
python train_simulate.py --idx 1 --data_path ../../DataSet --dataset CAVE --epochs 1000 --train_size 64 --stride 32 --batch_size 16 --lr_decay --save_freq 50 --device 7
```

## 🔵 ICVL
```
python train_simulate.py --idx 2 --data_path ../../DataSet --dataset ICVL --epochs 40 --train_size 64 --stride 64 --batch_size 16 --lr_decay --save_freq 5 --device 7
```

## 🟠 realworld dataset
```
python train_realworld.py --idx 3 --data_path ../../DataSet --dataset real_world --epochs 10 --train_size 64 --stride 64 --batch_size 16 --lr_decay --save_freq 1 --visual --visual_freq 1 --device 6
```

# 2. Generate
## 🟢 CAVE
### generate all
```
python generate.py --idx 1 --simulate --mosaic_save --pan_save --demosaic_save --gt_save --data_path ../../DataSet/ --dataset CAVE --load_model ./1/model/best_958.pth
```

### generate single
```
python generate.py --idx 1 --simulate --mosaic_save --pan_save --demosaic_save --gt_save --data_path ../../DataSet/ --dataset CAVE --load_model ./1/model/best_958.pth --data_id oil_painting_ms.mat
```

## 🔵 ICVL
### generate all
```
python generate.py --idx 1 --simulate --mosaic_save --pan_save --demosaic_save --gt_save --data_path ../../DataSet/ --dataset ICVL --load_model ./2/model/best_32.pth
```

### generate single
```
python generate.py --idx 1 --simulate --mosaic_save --pan_save --demosaic_save --gt_save --data_path ../../DataSet/ --dataset ICVL --load_model ./2/model/best_32.pth --data_id nachal_0823-1127.mat
```

## 🟠 RealWorld
### generate all
```
python generate.py --idx 1 --real_world --mosaic_save --pan_save --demosaic_save --data_path ../../DataSet/ --dataset real_world --load_model ./3/model/best_1.pth
```

### generate single
```
python generate.py --idx 1 --real_world --mosaic_save --pan_save --demosaic_save --data_path ../../DataSet/ --dataset real_world --load_model ./3/model/best_1.pth --data_id 1.raw 16.raw 37.raw
```

# 3. test
## 🟢 Cave
### test all
```
python test.py --idx 1 --simulate --data_path ./CAVE/1/result/mat/
```

### test single
```
python test.py --idx 1 --simulate --data_path ./CAVE/1/result/mat/ --data_id oil_painting_ms.mat
```
## 🔵 ICVL
### test all
```
python test.py --idx 1 --simulate --data_path ./ICVL/1/result/mat/
```

### test single
```
python test.py --idx 1 --simulate --data_path ./ICVL/1/result/mat/ --data_id nachal_0823-1127.mat
```
## 🟠 RealWorld
### test all
```
python test.py --idx 1 --real_world --data_path ./real_world/1/result/mat/ --patch_size 64
```

### test single
```
python test.py --idx 1 --real_world --data_path ./real_world/1/result/mat/ --patch_size 64 --data_id 3.mat
```

# 4. Visualize
## 🟢 CAVE
### visualize fused
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/fused/ --save_path ./CAVE/1/result/rgb/fused/ --data_type fused --data_id oil_painting_ms.mat --detach --detach_size 25 25 --detach_coordinate 80 170 80 170 --boxcolor b --boxwidth 1
```

### visualize diffmap
```
python visualize.py --visual_task diffmap --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/fused/ --mat_path_for_diff ./CAVE/1/result/mat/gt/ --save_path ./CAVE/1/result/rgb/diffmap/ --data_id oil_painting_ms.mat --detach --detach_size 25 25 --detach_coordinate 80 170 80 170 --boxcolor b --boxwidth 1 --mae_level 64 --sam_level 64
```

### visualize pan
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/pan/ --save_path ./CAVE/1/result/rgb/pan/ --data_type pan --data_id oil_painting_ms.mat --detach --detach_size 25 25 --detach_coordinate 80 170 80 170 --boxcolor b --boxwidth 1
```

### visualize gt
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/gt/ --save_path ./CAVE/1/result/rgb/gt/ --data_type mosaic --data_id oil_painting_ms.mat --detach --detach_size 25 25 --detach_coordinate 80 170 80 170 --boxcolor b --boxwidth 1
```

### visualize upmosaic
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/mosaic/ --save_path ./CAVE/1/result/rgb/upmosaic/ --data_id oil_painting_ms.mat --detach --detach_size 25 25 --detach_coordinate 80 170 80 170 --boxcolor b --boxwidth 1
```

## 🔵 ICVL
### visualize fused
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./ICVL/1/result/mat/fused/ --save_path ./ICVL/1/result/rgb/fused/ --data_type fused --data_id nachal_0823-1127.mat --detach --detach_size 64 64 --detach_coordinate 260 610 260 610 --boxcolor b --boxwidth 3
```

### visualize diffmap
```
python visualize.py --visual_task diffmap --spatial_ratio 8 --mat_path ./ICVL/1/result/mat/fused/ --mat_path_for_diff ./ICVL/1/result/mat/gt/ --save_path ./ICVL/1/result/rgb/diffmap/ --data_id nachal_0823-1127.mat --detach --detach_size 64 64 --detach_coordinate 260 610 260 610 --boxcolor b --boxwidth 3 --mae_level 16 --sam_level 16
```

### visualize pan
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./ICVL/1/result/mat/pan/ --save_path ./ICVL/1/result/rgb/pan/ --data_type pan --data_id nachal_0823-1127.mat --detach --detach_size 128 128 --detach_coordinate 300 600 100 160 --boxcolor r
```

### visualize gt
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./ICVL/1/result/mat/gt/ --save_path ./ICVL/1/result/rgb/gt/ --data_type mosaic --data_id nachal_0823-1127.mat --detach --detach_size 64 64 --detach_coordinate 260 610 260 610 --boxcolor b --boxwidth 3
```

### visualize upmosaic
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./ICVL/1/result/mat/mosaic/ --save_path ./ICVL/1/result/rgb/upmosaic/ --data_type upmosaic --data_id Lehavim_0910-1630.mat --detach --detach_size 128 128 --detach_coordinate 300 600 100 160 --boxcolor r
```

## 🟠 real-world
### visualize fused
#### img1
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/fused/ --save_path ./real_world/1/result/rgb/fused/ --data_id 1.mat --data_type fused --detach --detach_size 128 128 --detach_coordinate 260 690 260 690 --boxwidth 5 --boxcolor b
```
#### img2
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/fused/ --save_path ./real_world/1/result/rgb/fused/ --data_id 16.mat --data_type fused --detach --detach_size 128 128 --detach_coordinate 770 1880 770 1880 --boxwidth 5 --boxcolor b
```
#### img3
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/fused/ --save_path ./real_world/1/result/rgb/fused/ --data_id 37.mat --data_type fused --detach --detach_size 128 128 --detach_coordinate 690 470 690 470 --boxwidth 5 --boxcolor b
```

### visualize pan
#### img1
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/pan/ --save_path ./real_world/1/result/rgb/pan/ --data_id 1.mat --data_type pan --detach --detach_size 128 128 --detach_coordinate 260 690 260 690 --boxwidth 5 --boxcolor b
```
#### img2
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/pan/ --save_path ./real_world/1/result/rgb/pan/ --data_id 1.mat --data_type pan --detach --detach_size 128 128 --detach_coordinate 770 1880 770 1880 --boxwidth 5 --boxcolor b
```
#### img3
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/pan/ --save_path ./real_world/1/result/rgb/pan/ --data_id 1.mat --data_type pan --detach --detach_size 128 128 --detach_coordinate 690 470 690 470 --boxwidth 5 --boxcolor b
```

### visualize upmosaic
#### img1
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/mosaic/ --save_path ./real_world/1/result/rgb/upmosaic/ --data_id 1.mat --data_type upmosaic --detach --detach_size 128 128 --detach_coordinate 260 690 260 690 --boxwidth 5 --boxcolor b
```
#### img2
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/mosaic/ --save_path ./real_world/1/result/rgb/upmosaic/ --data_id 1.mat --data_type upmosaic --detach --detach_size 128 128 --detach_coordinate 770 1880 770 1880 --boxwidth 5 --boxcolor b
```
#### img3
```
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/mosaic/ --save_path ./real_world/1/result/rgb/upmosaic/ --data_id 1.mat --data_type upmosaic --detach_size 128 128 --detach_coordinate 690 470 690 470 --boxwidth 5 --boxcolor b
```

