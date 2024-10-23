
import numpy as np
import torch
from PIL import Image

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

