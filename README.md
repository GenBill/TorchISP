# TorchISP

This is the English version of the documentation. For the Chinese version, please refer to [README_CN.md](README_CN.md).

这是英文版文档。如需中文版，请参阅 [README_CN.md](README_CN.md)。

## Overview

TorchISP is an open-source library built on PyTorch, designed to convert 4-channel RGGB images into standard RGB images. It is suitable for various image processing and computer vision tasks. The library offers a flexible API, making it easy to integrate and extend.

## Features

- Converts 4-channel RGGB input to standard RGB output
- Efficient computation with PyTorch support and gradient backpropagation
- Simple API for quick adoption and integration

## Installation

To install the required dependency `pytorch-debayer`：

```bash
pip install git+https://github.com/cheind/pytorch-debayer
```

To install `TorchISP`:
```bash
pip install git+https://github.com/GenBill/TorchISP.git
```


## Quick Start
```python
import torch
from torchisp import RGGB2RGB

device = 'cpu'
rggb2rgb = RGGB2RGB(device=device)

# Input 4-channel RGGB image
rggb_img = torch.randn(1, 4, 256, 256).to(device)

# Convert to RGB image
rgb_img = rggb2rgb(rggb_img)

print(rgb_img.shape)
```
