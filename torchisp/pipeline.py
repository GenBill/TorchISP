

import torch
import torch.nn as nn
from debayer import Debayer5x5
from torchisp.module import SimpleCCM, GrayWorldWB

class ISP(nn.Module):
    def __init__(self, dgain=1.0, device='cuda', 
        wbcFunc=None, Debayer=None, ccm_matrix=None, ToneMapping=None, gamma=2.2
    ) -> None:
        super().__init__()
        self.device = device
        self.dgain = dgain
        self.r_gain, self.b_gain = None, None
        self.pixelshuffle = nn.PixelShuffle(2)

        self.wbcFunc = GrayWorldWB(device) if wbcFunc is None else wbcFunc
        
        self.Debayer = Debayer5x5().to(device) if Debayer is None else Debayer

        self.ccmFunc = SimpleCCM(device, ccm_matrix)

        self.gamma = gamma

    def forward(self, rggb):

        rggb2 = self.wbcFunc(rggb, self.r_gain, self.b_gain)
        
        # Debayer
        bayer = self.pixelshuffle(rggb2)
        rgb = self.Debayer(bayer).clamp(0.0, 1.0)
        
        # CCM
        rgb = self.ccmFunc(rgb).clamp(0.0, 1.0)
        
        # Gamma
        rgb = torch.pow(rgb, 1.0/self.gamma)
        
        return rgb
