import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RGB2RGGB(nn.Module):
    def __init__(self, rgb_gain=(2.0, 1.0, 2.0), device='cuda'):
        super().__init__()
        self.rgb_gain = torch.tensor(rgb_gain, device=device)
        self.device = device
        self.ccm_inv = torch.tensor(
            [[1.906, -0.422, -0.477], 
             [-0.516, 2.0, -0.484], 
             [-0.063, -0.852, 1.914]], device=device).inverse()

    def seq_rgb_inv_isp_2_rggb(self, batch_rgb_01: torch.Tensor):
        # Removed unnecessary constant and simplified the code
        batch_rgb_01 = batch_rgb_01 ** 2.2
        
        # Utilize batch processing to avoid Python loop over batch size
        batch_rggb = self.seq_rgb_2_rggb(batch_rgb_01)
        
        return batch_rggb

    def seq_rgb_2_rggb(self, batch_linear_rgb: torch.Tensor):
        # Apply inverse CCM using batch matrix multiplication
        batch_linear_rgb = torch.matmul(batch_linear_rgb.permute(0, 2, 3, 1), self.ccm_inv.T).permute(0, 3, 1, 2)
        
        # Upsample using bilinear interpolation
        batch_linear_rgb = F.interpolate(batch_linear_rgb, scale_factor=2.0, mode='bilinear')
        
        # Random offsets for downsampling to simulate the Bayer pattern
        scale = 2
        stride = 2 * scale
        dx, dy = np.random.randint(0, scale, size=2)
        
        # Extract RGGB channels
        R = batch_linear_rgb[:, 0:1, dx::stride, dy::stride] / self.rgb_gain[0]
        Gr = batch_linear_rgb[:, 1:2, dx::stride, dy+scale::stride] / self.rgb_gain[1]
        Gb = batch_linear_rgb[:, 1:2, dx+scale::stride, dy::stride] / self.rgb_gain[1]
        B = batch_linear_rgb[:, 2:, dx+scale::stride, dy+scale::stride] / self.rgb_gain[2]
        
        # Concatenate along channel dimension to form RGGB image
        rggb = torch.cat((R, Gr, Gb, B), dim=1)
        
        return rggb

    def forward(self, img: torch.Tensor):
        # Modified to directly use optimized batch processing functions
        # Ensure input has shape (N, 3, H, W)
        if img.dim() != 4 or img.size(1) != 3:
            raise ValueError("Input must have shape (N, 3, H, W)")
        
        raw = self.seq_rgb_inv_isp_2_rggb(img)
        return raw.clamp(0, 1)
    