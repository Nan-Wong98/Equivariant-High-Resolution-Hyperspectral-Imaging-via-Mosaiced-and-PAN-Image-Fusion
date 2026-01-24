# Equivariant High-Resolution Hyperspectral Imaging via Mosaiced and PAN Image Fusion (EFN)

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](这里放你的论文链接)

Official PyTorch implementation of the paper: **"Equivariant High-Resolution Hyperspectral Imaging via Mosaiced and PAN Image Fusion (EFN)"**.

## 📖 Introduction

In this repository, we provide the code for our proposed **Equivariant Fusion Network (EFN)** along with implementations of **12 state-of-the-art (SOTA) competing methods**. 

Our goal is to provide a comprehensive benchmark for mosaiced and PAN image fusion. All methods share a **unified framework** and consistent execution logic, making it easy for researchers to reproduce results and compare performance.

## 📂 Project Structure

This repository contains **13 individual projects** (sub-folders), including:

- **Proposed Method:** `EFN` (Equivariant Fusion Network)
- **Competing Methods:** `PPID_PanGAN`, `PPID_VBPN`, `PPID_WFANet`, `SpNet_PanGAN`, `SpNet_VBPN`, `SpNet_WFANet`, `SFNet_PanGAN`, `SFNet_VBPN`, `SFNet_WFANet`, `LSAN_PanGAN`, `LSAN_VBPN`, `LSAN_WFANet`.

### Unified Workflow
Each project folder follows the exact same file structure and logic:

| File Name | Function |
| :--- | :--- |
| `GetDataSet.py` | 🛠 **Data Preparation:** Generates training/testing data from raw datasets. |
| `train.py` | 🚀 **Training:** Trains the model. |
| `generate.py` | 💾 **Inference:** Generates fusion results using pretrained weights. |
| `test.py` | 📊 **Evaluation:** Calculates quantitative metrics (PSNR, SAM, ERGAS, Q2n, QNR etc.). |
| `visualize.py` | 🎨 **Visualization:** Visualizes the generated HSI and MAE map results. |

## 📦 Pretrained Weights
Pretrained weights of all the fusion methods are packaged in [Release](https://github.com/Nan-Wong98/Equivariant-High-Resolution-Hyperspectral-Imaging-via-Mosaiced-and-PAN-Image-Fusion/releases).

## ⚙️ Requirements
*   h5py==3.15.1
*   Imath==0.0.2
*   matplotlib==3.10.8
*   numpy==2.4.1
*   opencv_python==4.13.0.90
*   OpenEXR==3.4.4
*   pytorch_msssim==1.0.0
*   scipy==1.17.0
*   torch==2.10.0+cu126
*   torchvision==0.25.0+cu126
*   tqdm==4.67.1

Install dependencies via:
```bash
pip install -r requirements.txt
````

## 📝 Citation

If you find this code or our dataset useful for your research, please verify strictly and cite our paper:
```
@article{EFN2026,
  author={Wang, Nan and Guo, Anjing and Dian, Renwei, and Li, Shutao},
  journal={IEEE Transactions on Image Processing},
  title={Equivariant High-Resolution Hyperspectral Imaging via Mosaiced and PAN Image Fusion},
  year={2026},
  doi={10.1109/TIP.2026.3657219}
}
```

## 📧 Contact
If any question, please contact with me.

E-mail: wangn@hnu.edu.cn

Wechat: wn343816504