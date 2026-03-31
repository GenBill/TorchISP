

import torch
import torch.nn as nn
from debayer import Debayer5x5
from torchisp.constants import NUMERIC_FLOOR
from torchisp.module import SimpleCCM, GrayWorldWB

class ISP(nn.Module):
    def __init__(self, dgain=1.0, device='cuda', 
        whitelevel=65535, blacklevel=4096,
        wbcFunc=None, Debayer=None, ccm_matrix=None, ToneMapping=None, gamma=2.2
    ) -> None:
        super().__init__()
        self.device = device
        self.whitelevel = whitelevel
        self.blacklevel = blacklevel
        
        self.dgain = dgain
        self.r_gain, self.b_gain = None, None
        self.pixelshuffle = nn.PixelShuffle(2)

        self.wbcFunc = GrayWorldWB(device) if wbcFunc is None else wbcFunc
        
        self.Debayer = Debayer5x5().to(device) if Debayer is None else Debayer

        self.ccmFunc = SimpleCCM(device, ccm_matrix)

        self.gamma = gamma

    def normalize(self, rggb):
        return (rggb - self.blacklevel) / (self.whitelevel - self.blacklevel)
    
    def denormalize(self, rggb):
        return rggb * (self.whitelevel - self.blacklevel) + self.blacklevel
    
    def forward(self, rggb, apply_normalize=True):
        """
        Args:
            rggb: RGGB tensor.
            apply_normalize: If True (default), apply ``(rggb - blacklevel) /
                (whitelevel - blacklevel)`` (sensor code range → [0, 1]).
                If False, skip this step because rggb is already in that
                normalized [0, 1] form (e.g. RGB2RGGB / InvISP optimization,
                or RawLoader.get_raw16).
        """
        if apply_normalize:
            rggb = self.normalize(rggb)
        rggb2 = self.wbcFunc(rggb, self.r_gain, self.b_gain).clamp(min=NUMERIC_FLOOR)

        # Debayer
        bayer = self.pixelshuffle(rggb2)
        rgb = self.Debayer(bayer).clamp(NUMERIC_FLOOR, 1.0)
        
        # CCM
        rgb = self.ccmFunc(rgb).clamp(NUMERIC_FLOOR, 1.0)
        
        # Gamma
        rgb = torch.pow(rgb, 1.0/self.gamma)
        
        return rgb
