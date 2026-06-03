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
rggb2rgb = ISP(device=device, whitelevel=65535, blacklevel=4096)

# Input 4-channel RGGB image
rggb_img = (torch.rand(1, 4, 256, 256) * (65535 - 4096) + 4096).to(device)
# rggb_img = RawLoader()('your_raw_saved_as_uint16_numpy_bin').to(device)

# Convert to RGB image
rgb_img = rggb2rgb(rggb_img)

print(rgb_img.shape)
```


## Demosaic modes

TorchISP supports multiple demosaic modes through the `ISP` pipeline. The input remains the existing packed 4-channel Bayer tensor format. For the default `bayer_pattern="RGGB"`, channels are interpreted as `R, G1, G2, B` in row-major 2x2 tile order.

- `debayer5x5`: default PyTorch demosaic backend. This preserves the original behavior and remains differentiable.
- `amaze`: high-quality display-oriented AMaZE demosaic backend. This mode is intended for visual preview and presentation images and requires an optional tensor-capable RawTherapee/librtprocess Python binding. TorchISP does not silently fall back to OpenCV when this mode is requested.
- `rcd`: optional high-quality RCD demosaic backend. Like `amaze`, this requires an optional tensor-capable librtprocess binding.
- `opencv_ea`: optional OpenCV Edge-Aware fallback backend. Install `opencv-python` or `opencv-python-headless` to use it.

The default mode remains `debayer5x5`, so existing code using `ISP(...)` is unchanged.

```python
from torchisp import ISP

# Default behavior, same as before
isp = ISP()

# Explicit default PyTorch backend
isp = ISP(demosaic_mode="debayer5x5")

# High-quality preview backend, if a compatible librtprocess binding is installed
isp = ISP(demosaic_mode="amaze")

# Optional OpenCV Edge-Aware fallback
isp = ISP(demosaic_mode="opencv_ea")
```

### AMaZE/RCD optional dependency note

`demosaic_mode="amaze"` and `demosaic_mode="rcd"` are adapter hooks, not bundled demosaic implementations. To use either mode today, you must install or build a Python package named `librtprocess` that can operate on TorchISP's in-memory packed Bayer tensors after they are converted to a 2D Bayer mosaic. The package is expected to expose:

```python
amaze_demosaic(mosaic, bayer_pattern, max_value)
rcd_demosaic(mosaic, bayer_pattern, max_value)
```

TorchISP does not assume the input is a DNG/ARW/NEF file and does not invoke RawTherapee on a RAW file container. If this optional binding is unavailable, requesting `amaze` or `rcd` raises an error instead of silently falling back to OpenCV. Use `demosaic_mode="opencv_ea"` explicitly if the OpenCV Edge-Aware fallback is acceptable for your preview workflow.

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

