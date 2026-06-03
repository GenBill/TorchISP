

import warnings

import torch
import torch.nn as nn
from debayer import Debayer5x5
from torchisp.constants import NUMERIC_FLOOR
from torchisp.module import SimpleCCM, GrayWorldWB
from torchisp.demosaic import (
    DISPLAY_DEMOSAIC_MODES,
    SUPPORTED_BAYER_PATTERNS,
    create_display_demosaic,
)

class ISP(nn.Module):
    def __init__(self, dgain=1.0, device='cuda', 
        whitelevel=65535, blacklevel=4096,
        wbcFunc=None, Debayer=None, ccm_matrix=None, ToneMapping=None, gamma=2.2,
        demosaic_mode: str = "debayer5x5", bayer_pattern: str = "RGGB"
    ) -> None:
        super().__init__()
        self.device = device
        self.whitelevel = whitelevel
        self.blacklevel = blacklevel
        self.demosaic_mode = demosaic_mode.lower()
        self.bayer_pattern = bayer_pattern.upper()
        if self.bayer_pattern not in SUPPORTED_BAYER_PATTERNS:
            raise ValueError(
                f"Unsupported bayer_pattern: {bayer_pattern}. Supported patterns: "
                f"{', '.join(SUPPORTED_BAYER_PATTERNS)}"
            )
        if self.demosaic_mode not in ("debayer5x5", *DISPLAY_DEMOSAIC_MODES):
            raise ValueError(f"Unsupported demosaic_mode: {demosaic_mode}")
        
        self.dgain = dgain
        self.r_gain, self.b_gain = None, None
        self.pixelshuffle = nn.PixelShuffle(2)

        self.wbcFunc = GrayWorldWB(device) if wbcFunc is None else wbcFunc
        
        self.Debayer = None
        self.display_demosaic = None
        if self.demosaic_mode == "debayer5x5":
            self.Debayer = Debayer5x5().to(device) if Debayer is None else Debayer
        else:
            if Debayer is not None:
                warnings.warn(
                    "Debayer is ignored when demosaic_mode is not 'debayer5x5'.",
                    UserWarning,
                    stacklevel=2,
                )
            self.display_demosaic = create_display_demosaic(
                self.demosaic_mode,
                bayer_pattern=self.bayer_pattern,
                max_value=65535,
            )

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

        # Demosaic
        if self.demosaic_mode == "debayer5x5":
            bayer = self.pixelshuffle(rggb2)
            rgb = self.Debayer(bayer).clamp(NUMERIC_FLOOR, 1.0)
        elif self.display_demosaic is not None:
            rgb = self.display_demosaic(rggb2).clamp(NUMERIC_FLOOR, 1.0)
        else:
            raise ValueError(f"Unsupported demosaic_mode: {self.demosaic_mode}")
        
        # CCM
        rgb = self.ccmFunc(rgb).clamp(NUMERIC_FLOOR, 1.0)
        
        # Gamma
        rgb = torch.pow(rgb, 1.0/self.gamma)
        
        return rgb
