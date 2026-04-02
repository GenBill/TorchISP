
import numpy as np
import torch
from PIL import Image


def raw4_to_uint16_bayer(
    raw4,
    blacklevel=4096,
    whitelevel=65535,
    normalized=False,
):
    """
    将 RGGB 四通道 raw4 还原为 uint16 Bayer 平面 (H, W)，与 RawLoader.get_raw16 互逆。

    raw4: torch.Tensor，形状 (B, 4, H/2, W/2) 或 (4, H/2, W/2)。通常为 InvISP.forward 的输出
    （已为传感器计数 float，此时 normalized=False）。
    若为 RawLoader.get_raw16 的归一化输出 [0,1]，设 normalized=True；whitelevel 需与加载时一致
    （get_raw16 使用 65472 作为上端，应传 whitelevel=65472）。
    """
    x = raw4.detach().cpu().float() if isinstance(raw4, torch.Tensor) else torch.as_tensor(raw4).float()
    if x.dim() == 4:
        x = x[0]
    if x.shape[0] != 4:
        raise ValueError(f"raw4 通道数应为 4，当前为 {x.shape[0]}")
    if normalized:
        x = x * (float(whitelevel) - float(blacklevel)) + float(blacklevel)
    x = x.clamp(0, 65535).numpy()
    x = np.rint(x).astype(np.uint16)
    _, h2, w2 = x.shape
    rggb = np.transpose(x, (1, 2, 0)).reshape(h2, w2, 2, 2)
    rggb = np.transpose(rggb, (0, 2, 1, 3))
    return rggb.reshape(h2 * 2, w2 * 2)


def save_raw4_as_bayer(raw4, save_path, **kwargs):
    """将 raw4 转为 uint16 Bayer 并写入二进制文件（与 np.fromfile 读法一致）。"""
    bayer = raw4_to_uint16_bayer(raw4, **kwargs)
    bayer.tofile(save_path)


class RawLoader():
    def __init__(self, H, W, bl=4096):
        self.H = H
        self.W = W
        self.bl = bl
        
    def get_raw16(self, raw_path):
        raw = np.fromfile(raw_path, np.uint16).reshape(self.H, self.W).astype(np.float32)
        raw4 = raw.reshape(self.H//2, 2, self.W//2, 2).transpose(0,2,1,3).reshape(self.H//2, self.W//2, 4).transpose(2,0,1)
        raw4 = (torch.from_numpy(raw4) - self.bl) / (65472 - self.bl)
        return raw4.unsqueeze(0)

    def __call__(self, raw_path):
        return self.get_raw16(raw_path)

class RGBLoader():
    def __init__(self):
        pass
    
    def get_rgb(self, rgb_path):
        # 使用PIL读取RGB图像
        img = Image.open(rgb_path).convert('RGB')
        # 将图像转换为 numpy 数组并归一化为 [0, 1]
        img_np = np.array(img).astype(np.float32) / 255.0
        # 将 numpy 数组转换为 PyTorch 张量，调整维度为 [C, H, W]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        return img_tensor.unsqueeze(0)

    def __call__(self, rgb_path):
        return self.get_rgb(rgb_path)
    
# 示例用法
if __name__ == "__main__":
    # 假设我们有一个Bayer图像尺寸为4000x3000，黑电平值为4096
    bayer_loader = RawLoader(H=4000, W=3000, bl=4096)
    raw_tensor = bayer_loader.get_raw16("path/to/raw_file.raw")
    print("RAW Tensor shape:", raw_tensor.shape)

    # 读取RGB图像
    rgb_loader = RGBLoader()
    rgb_tensor = rgb_loader.get_rgb("path/to/rgb_image.jpg")
    print("RGB Tensor shape:", rgb_tensor.shape)

