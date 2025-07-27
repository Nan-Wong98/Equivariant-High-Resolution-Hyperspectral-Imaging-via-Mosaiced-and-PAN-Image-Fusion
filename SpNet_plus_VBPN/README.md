# Train
## CAVE
``` 
# demosaic
python train_demosaic_simulate.py --idx 1 --data_path ../../DataSet --dataset CAVE --epochs 10000 --train_size 64 --stride 32 --batch_size 1 --lr_decay --save_freq 50 --device 5

# pansharpening
python train_pansharpening_simulate.py --idx 2 --data_path ../../DataSet --dataset CAVE --epochs 100 --train_size 64 --stride 32 --batch_size 16  --resume_demosaic ./1/model/best_9757.pth --lr_decay --save_freq 5 --device 3
```

## ICVL
```
# demosaic
python train_demosaic_simulate.py --idx 3 --data_path ../../DataSet --dataset ICVL --epochs 200 --train_size 64 --stride 32 --batch_size 16 --lr_decay --save_freq 50 --device 4

# pansharpening
python train_pansharpening_simulate.py --idx 4 --data_path ../../DataSet --dataset ICVL --epochs 40 --train_size 64 --stride 64 --batch_size 16  --resume_demosaic ./3/model/best_130.pth --lr_decay --save_freq 5 --device 6
```

## realworld dataset
```
# demosaic
python train_demosaic_realworld.py --idx 5 --data_path ../DataSet --dataset real_world --epochs 1000 --train_size 64 --stride 32 --batch_size 16 --lr_decay --save_freq 50 --visual --visual_freq 50

# pansharpening
python train_pansharpening_realworld.py --idx 6 --data_path ../../DataSet --dataset real_world --epochs 10 --train_size 64 --stride 64 --batch_size 16  --resume_demosaic ./5/model/best_840.pth --lr_decay --save_freq 1 --visual --visual_freq 1 --device 7
```

# Generate
## CAVE
```
# all
python generate.py --idx 1 --simulate --mosaic_save --pan_save --demosaic_save --gt_save --data_path ../../DataSet/ --dataset CAVE --load_demosaic_model ./1/model/best_9757.pth --load_ps_model ./2/model/best_33.pth

# assign
python generate.py --idx 1 --simulate --mosaic_save --pan_save --demosaic_save --gt_save --data_path ../../DataSet/ --dataset CAVE --load_demosaic_model ./1/model/best_9757.pth --load_ps_model ./2/model/best_33.pth --data_id oil_painting_ms.mat
```

## ICVL
```
# all
python generate.py --idx 1 --simulate --mosaic_save --pan_save --demosaic_save --gt_save --data_path ../../DataSet/ --dataset ICVL --load_demosaic_model ./3/model/best_130.pth --load_ps_model ./4/model/best_40.pth

# assign
python generate.py --idx 1 --simulate --mosaic_save --pan_save --demosaic_save --gt_save --data_path ../../DataSet/ --dataset ICVL --load_demosaic_model ./3/model/best_130.pth --load_ps_model ./4/model/best_40.pth --data_id nachal_0823-1127.mat
```

## RealWorld
```
# all
python generate.py --idx 1 --real_world --mosaic_save --pan_save --demosaic_save --data_path ../../DataSet/ --dataset real_world --load_demosaic_model ./5/model/best_840.pth --load_ps_model ./6/model/best_9.pth

# assign
python generate.py --idx 1 --real_world --mosaic_save --pan_save --demosaic_save --data_path ../../DataSet/ --dataset real_world --load_demosaic_model ./5/model/best_840.pth --load_ps_model ./6/model/best_9.pth --data_id 1.raw 16.raw 37.raw
```

# test
## Cave
```
# test all
python test.py --idx 1 --simulate --data_path ./CAVE/1/result/mat/

# test assign
python test.py --idx 1 --simulate --data_path ./CAVE/1/result/mat/ --data_id oil_painting_ms.mat
```
## ICVL
```
# test all
python test.py --idx 1 --simulate --data_path ./ICVL/1/result/mat/

# test assign
python test.py --idx 1 --simulate --data_path ./ICVL/1/result/mat/ --data_id nachal_0823-1127.mat
```
## RealWorld
```
# test all
python test.py --idx 1 --real_world --data_path ./real_world/1/result/mat/ --patch_size 64

# test assign
python test.py --idx 1 --real_world --data_path ./real_world/1/result/mat/ --patch_size 64 --data_id 3.mat
```

# Visualize
## simulate
```
# Visualize fused
# CAVE
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/fused/ --save_path ./CAVE/1/result/rgb/fused/ --data_type fused --data_id oil_painting_ms.mat --detach --detach_size 25 25 --detach_coordinate 80 170 80 170 --boxcolor b --boxwidth 1

# ICVL
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./ICVL/1/result/mat/fused/ --save_path ./ICVL/1/result/rgb/fused/ --data_type fused --data_id nachal_0823-1127.mat --detach --detach_size 64 64 --detach_coordinate 260 610 260 610 --boxcolor b --boxwidth 3

# Visualize diffmap
# CAVE
python visualize.py --visual_task diffmap --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/fused/ --mat_path_for_diff ./CAVE/1/result/mat/gt/ --save_path ./CAVE/1/result/rgb/diffmap/ --data_id oil_painting_ms.mat --detach --detach_size 25 25 --detach_coordinate 80 170 80 170 --boxcolor b --boxwidth 1 --mae_level 64 --sam_level 64

# ICVL
python visualize.py --visual_task diffmap --spatial_ratio 8 --mat_path ./ICVL/1/result/mat/fused/ --mat_path_for_diff ./ICVL/1/result/mat/gt/ --save_path ./ICVL/1/result/rgb/diffmap/ --data_id nachal_0823-1127.mat --detach --detach_size 64 64 --detach_coordinate 260 610 260 610 --boxcolor b --boxwidth 3 --mae_level 16 --sam_level 16

# Visualize pan
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/pan/ --save_path ./CAVE/1/result/rgb/pan/ --data_type pan --data_id oil_painting_ms.mat --detach --detach_size 25 25 --detach_coordinate 80 170 80 170 --boxcolor b --boxwidth 1
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./ICVL/1/result/mat/pan/ --save_path ./ICVL/1/result/rgb/pan/ --data_type pan --data_id nachal_0823-1127.mat --detach --detach_size 128 128 --detach_coordinate 300 600 100 160 --boxcolor r

# Visualize gt
# CAVE
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/gt/ --save_path ./CAVE/1/result/rgb/gt/ --data_type mosaic --data_id oil_painting_ms.mat --detach --detach_size 25 25 --detach_coordinate 80 170 80 170 --boxcolor b --boxwidth 1

# ICVL
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./ICVL/1/result/mat/gt/ --save_path ./ICVL/1/result/rgb/gt/ --data_type mosaic --data_id nachal_0823-1127.mat --detach --detach_size 64 64 --detach_coordinate 260 610 260 610 --boxcolor b --boxwidth 3

# Visualize upmosaic
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/mosaic/ --save_path ./CAVE/1/result/rgb/upmosaic/ --data_id oil_painting_ms.mat --detach --detach_size 25 25 --detach_coordinate 80 170 80 170 --boxcolor b --boxwidth 1
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./ICVL/1/result/mat/mosaic/ --save_path ./ICVL/1/result/rgb/upmosaic/ --data_type upmosaic --data_id nachal_0823-1127.mat --detach --detach_size 128 128 --detach_coordinate 300 600 100 160 --boxcolor r
```

## real-world
```
# Visualizing fused detach
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/fused/ --save_path ./real_world/1/result/rgb/fused/ --data_id 1.mat --data_type fused --detach --detach_size 128 128 --detach_coordinate 260 690 260 690 --boxwidth 5 --boxcolor b
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/fused/ --save_path ./real_world/1/result/rgb/fused/ --data_id 16.mat --data_type fused --detach --detach_size 128 128 --detach_coordinate 770 1880 770 1880 --boxwidth 5 --boxcolor b
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/fused/ --save_path ./real_world/1/result/rgb/fused/ --data_id 37.mat --data_type fused --detach --detach_size 128 128 --detach_coordinate 690 470 690 470 --boxwidth 5 --boxcolor b

# Visualize pan real-world detach
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/pan/ --save_path ./real_world/1/result/rgb/pan/ --data_id 1.mat --data_type pan --detach --detach_size 128 128 --detach_coordinate 260 690 260 690 --boxwidth 5 --boxcolor b
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/pan/ --save_path ./real_world/1/result/rgb/pan/ --data_id 1.mat --data_type pan --detach --detach_size 128 128 --detach_coordinate 770 1880 770 1880 --boxwidth 5 --boxcolor b
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/pan/ --save_path ./real_world/1/result/rgb/pan/ --data_id 1.mat --data_type pan --detach --detach_size 128 128 --detach_coordinate 690 470 690 470 --boxwidth 5 --boxcolor b

# Visualize upmosaic real-world detach
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/mosaic/ --save_path ./real_world/1/result/rgb/upmosaic/ --data_id 1.mat --data_type upmosaic --detach --detach_size 128 128 --detach_coordinate 260 690 260 690 --boxwidth 5 --boxcolor b
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/mosaic/ --save_path ./real_world/1/result/rgb/upmosaic/ --data_id 1.mat --data_type upmosaic --detach --detach_size 128 128 --detach_coordinate 770 1880 770 1880 --boxwidth 5 --boxcolor b
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/mosaic/ --save_path ./real_world/1/result/rgb/upmosaic/ --data_id 1.mat --data_type upmosaic --detach_size 128 128 --detach_coordinate 690 470 690 470 --boxwidth 5 --boxcolor b
```