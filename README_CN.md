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
rggb2rgb = ISP(device=device)

# Input 4-channel RGGB image
rggb_img = torch.randn(1, 4, 256, 256).to(device)
# rggb_img = RawLoader()('your_raw_saved_as_uint16_numpy_bin').to(device)

# Convert to RGB image
rgb_img = rggb2rgb(rggb_img)

print(rgb_img.shape)
```


## Inverse ISP
```python
import torch
from torchisp import RGGB2RGB
from torchisp import RawLoader, ISP, InvISP

device = 'cuda'
rgb_path = 'rawdata/lsdir_1000.png'
rgb_img = RGBLoader()(rgb_path).to(device)

rggb2rgb = ISP(device=device)
# Recommended to fix wbgain for stable effect
rggb2rgb.r_gain, rggb2rgb.b_gain = 2.0, 2.0

loss_fn = nn.L1Loss() # nn.MSELoss()
inv_isp = InvISP(loss_fn, rggb2rgb, device=device,
    lr = 1e-4, 
    nb_iter = 16000,
    eps_iter = 16 / 255,
)

rggb_img = inv_isp(rgb_img)
rgb_img2 = rggb2rgb(rggb_img)

save_image(rgb_img2, 'outputs/lsdir_1000_output.png')

```


## Reference  
Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks," International Conference on Learning Representations (ICLR), 2018.

