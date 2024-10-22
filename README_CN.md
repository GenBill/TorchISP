# TorchISP

This is the Chinese version of the documentation. For the English version, please refer to [README.md](README.md).

这是中文版文档。如需英文版，请参阅 [README.md](README.md)。

## 简介

TorchISP 是一个基于 PyTorch 的开源库，用于从 4 通道 RGGB 图像转换为标准 RGB 图像。适用于各种图像处理和计算机视觉任务。该库提供了灵活的 API 接口，便于集成和二次开发。

## 主要功能

- 输入 4 通道 RGGB 图像，输出标准 RGB 图像
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

device = 'cpu'
rggb2rgb = RGGB2RGB(device=device)

# 输入 4 通道 RGGB 图像
rggb_img = torch.randn(1, 4, 256, 256).to(device)

# 转换为 RGB 图像
rgb_img = rggb2rgb(rggb_img)

print(rgb_img.shape)
```
