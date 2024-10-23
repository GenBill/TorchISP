# TorchISP

This is the English version of the documentation. For the Chinese version, please refer to [README_CN.md](README_CN.md).

这是英文版文档。如需中文版，请参阅 [README_CN.md](README_CN.md)。

## Overview

TorchISP is an open-source library built on PyTorch, designed to convert 4-channel RGGB images into standard RGB images. It is suitable for various image processing and computer vision tasks. The library offers a flexible API, making it easy to integrate and extend.

## Features

- Converts 4-channel RGGB input to standard RGB output
- Inverse ISP converts standard RGB input to 4-channel RGGB via PGD adverserial attack
- Efficient computation with PyTorch support and gradient backpropagation
- Simple API for quick adoption and integration

## Installation

To install the required dependency `pytorch-debayer`：

```bash
pip install git+https://github.com/cheind/pytorch-debayer
```

To install `TorchISP`:
```bash
pip install torchisp
```


## Quick Start
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

