# TorchISP

This is the Chinese version of the documentation. For the English version, please refer to [README.md](README.md).

这是中文版文档。如需英文版，请参阅 [README.md](README.md)。

## 简介

TorchISP 是一个基于 PyTorch 的开源库，用于从 4 通道 RGGB 图像转换为标准 RGB 图像。适用于各种图像处理和计算机视觉任务。该库提供了灵活的 API 接口，便于集成和二次开发。

## 主要功能

- 输入 4 通道 RGGB 图像，输出标准 RGB 图像
- 通过 PGD 攻击逆转 ISP 通路，输入标准 RGB 图像，输出 4 通道 RGGB 图像
- 兼容 PyTorch 的高效计算和梯度回传
- 简洁的 API 接口，便于快速上手和集成

## 安装

安装必要依赖库 `pytorch-debayer`：

```bash
pip install git+https://github.com/cheind/pytorch-debayer
```

安装 `TorchISP`：
```bash
pip install torchisp
```


## 快速开始
```python
import torch
from torchisp import RGGB2RGB
from torchisp import RawLoader, ISP, InvISP

device = 'cpu'
# rggb2rgb = RGGB2RGB(device=device)
rggb2rgb = ISP(device=device, whitelevel=65535, blacklevel=4096)

# Input 4-channel RGGB image
rggb_img = (torch.rand(1, 4, 256, 256) * (65535 - 4096) + 4096).to(device)
# rggb_img = RawLoader()('your_raw_saved_as_uint16_numpy_bin').to(device)

# Convert to RGB image
rgb_img = rggb2rgb(rggb_img)

print(rgb_img.shape)
```


## Demosaic modes（去马赛克模式）

TorchISP 现在可以在 `ISP` pipeline 中选择不同的 demosaic mode。输入仍然是现有的 4 通道 Bayer packed tensor。默认 `bayer_pattern="RGGB"` 时，通道顺序解释为 2x2 tile 的 row-major 顺序：`R, G1, G2, B`。

- `debayer5x5`：默认 PyTorch demosaic backend，保持原有行为，并且仍然适合需要 torch 流程/梯度的场景。
- `amaze`：面向高质量视觉预览/展示图的 AMaZE demosaic adapter。该模式不是内置实现，需要额外安装或自行编译/封装 tensor-capable 的 RawTherapee/librtprocess Python binding。请求该模式时，TorchISP 不会静默 fallback 到 OpenCV。
- `rcd`：可选的高质量 RCD demosaic adapter，与 `amaze` 一样需要额外的 tensor-capable librtprocess Python binding。
- `opencv_ea`：可选 OpenCV Edge-Aware fallback backend。使用前需安装 `opencv-python` 或 `opencv-python-headless`。

默认模式仍然是 `debayer5x5`，所以现有 `ISP(...)` 调用保持不变。

```python
from torchisp import ISP

# 默认行为，与之前相同
isp = ISP()

# 显式使用默认 PyTorch backend
isp = ISP(demosaic_mode="debayer5x5")

# 高质量预览 backend；需要 compatible librtprocess binding
isp = ISP(demosaic_mode="amaze")

# 可选 OpenCV Edge-Aware fallback
isp = ISP(demosaic_mode="opencv_ea")
```

### AMaZE/RCD 可选依赖说明

是的，当前 `demosaic_mode="amaze"` / `demosaic_mode="rcd"` 需要你安装或自行编译/封装一个名为 `librtprocess` 的 Python package，并且它必须能直接处理 TorchISP 的内存 tensor 数据转换出的 2D Bayer mosaic。TorchISP 期望该 package 暴露以下函数：

```python
amaze_demosaic(mosaic, bayer_pattern, max_value)
rcd_demosaic(mosaic, bayer_pattern, max_value)
```

TorchISP 的输入不是 DNG/ARW/NEF 等 RAW 文件容器，因此这里不会假设有 RAW 文件路径，也不会调用 RawTherapee 去打开文件。如果没有安装这个可选 binding，请求 `amaze` 或 `rcd` 会抛出清晰错误，而不是静默退回 OpenCV。若能接受 OpenCV Edge-Aware 的质量，请显式使用 `demosaic_mode="opencv_ea"`。


## Inverse ISP
```python
import torch
from torchisp import RGGB2RGB
from torchisp import RawLoader, ISP, InvISP

device = 'cuda'
rgb_path = 'rawdata/lsdir_1000.png'
rgb_img = RGBLoader()(rgb_path).to(device)

rggb2rgb = ISP(device=device, whitelevel=65535, blacklevel=4096)
# Recommended to fix wbgain for stable effect
rggb2rgb.r_gain, rggb2rgb.b_gain = 2.0, 2.0

loss_fn = nn.L1Loss() # nn.MSELoss()
inv_isp = InvISP(loss_fn, rggb2rgb, device=device,
    lr = 1e-4, 
    nb_iter = 1000,
    eps_iter = 2/255, 
    whitelevel=65535, blacklevel=4096
)

rggb_img = inv_isp(rgb_img)
rgb_img2 = rggb2rgb(rggb_img)

save_image(rgb_img2, 'outputs/lsdir_1000_output.png')

```


## Reference  
Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks," International Conference on Learning Representations (ICLR), 2018.

