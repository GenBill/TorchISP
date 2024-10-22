
import torch
import torch.nn as nn
from debayer import Debayer5x5

class RGGB2RGB(nn.Module):
    def __init__(self, dgain=1.0, device='cuda') -> None:
        super().__init__()
        self.device = device
        self.dgain = dgain
        self.r_gain, self.b_gain = None, None
        self.pixelshuffle = nn.PixelShuffle(2)
        self.Debayer = Debayer5x5().to(device)
        # self.Debayer.eval()
    
    def get_default_wb(self, rggb):
        mean_r = rggb[:, 0, :, :].mean()
        mean_g = (rggb[:, 1, :, :].mean() + rggb[:, 2, :, :].mean()) / 2
        mean_b = rggb[:, 3, :, :].mean()
        return mean_g / mean_r, mean_g / mean_b
    
    def load_default_wb(self, rggb):
        mean_r = rggb[:, 0, :, :].mean()
        mean_g = (rggb[:, 1, :, :].mean() + rggb[:, 2, :, :].mean()) / 2
        mean_b = rggb[:, 3, :, :].mean()
        self.r_gain, self.b_gain = mean_g / mean_r, mean_g / mean_b

    def wb_gain(self, rggb, r_gain=2.0, b_gain=2.0):
        rggb2 = rggb.clone()
        rggb2[:, 0, :, :] = r_gain * rggb[:, 0, :, :]
        rggb2[:, 3, :, :] = b_gain * rggb[:, 3, :, :]
        return rggb2

    def ccm(self, rgb):
        # ccm_tensor = torch.tensor(
        #     [[1.3, -0.2, -0.1],
        #     [-0.1, 1.3, -0.2], 
        #     [0.1, -0.4, 1.3]], device=self.device).transpose(0,1)
        ccm_tensor = torch.tensor(
            [[1.906, -0.422, -0.477], 
             [-0.516, 2.0, -0.484], 
             [-0.063, -0.852, 1.914]], device=self.device).transpose(0,1)
        rgb = torch.matmul(rgb.permute(0, 2, 3, 1), ccm_tensor)
        return rgb.permute(0, 3, 1, 2)
    
    def forward(self, rggb):
        if self.r_gain is None or self.b_gain is None:
            with torch.no_grad():
                r_gain, b_gain = self.get_default_wb(rggb)
        else:
            r_gain, b_gain = self.r_gain, self.b_gain
        
        rggb2 = self.wb_gain(rggb*self.dgain, r_gain, b_gain).clamp(0.0, 1.0)
        bayer = self.pixelshuffle(rggb2)
        rgb = self.Debayer(bayer).clamp(0.0, 1.0)
        
        rgb = self.ccm(rgb).clamp(0.0, 1.0)
        rgb = torch.pow(rgb, 1/2.2)
        
        return rgb
