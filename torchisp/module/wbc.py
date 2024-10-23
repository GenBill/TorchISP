import torch

class BasicWB(torch.nn.Module):

    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
    
    def get_wbgain(self, inp_raw: torch.Tensor) -> tuple:
        return 2.0, 2.0
    
    def forward(self, rggb, r_gain=None, b_gain=None):
        if r_gain is None or b_gain is None:
            # with torch.no_grad(): 
            #     r_gain, b_gain = self.get_wbgain(rggb)
            r_gain, b_gain = self.get_wbgain(rggb)
        
        rggb2 = rggb.clone()
        rggb2[:, 0, :, :] = r_gain * rggb[:, 0, :, :]
        rggb2[:, 3, :, :] = b_gain * rggb[:, 3, :, :]
        return rggb2


class GrayWorldWB(BasicWB):

    def __init__(self, device='cpu'):
        super().__init__(device)
    
    def get_wbgain(self, inp_raw: torch.Tensor) -> tuple:
        """
        Args:
            inp_raw: (N, 4, H, W) tensor representing RGGB RAW data
        Returns:
            rgain: (N,) tensor
            bgain: (N,) tensor
        """
        assert inp_raw.shape[1] == 4 and inp_raw.dim() == 4, "Input should have shape (N, 4, H, W)"

        # Separate R, G1, G2, B channels
        R = inp_raw[:, 0, :, :]
        G1 = inp_raw[:, 1, :, :]
        G2 = inp_raw[:, 2, :, :]
        B = inp_raw[:, 3, :, :]

        # Calculate channel averages
        R_mean = R.mean(dim=(1, 2))
        G_mean = (G1.mean(dim=(1, 2)) + G2.mean(dim=(1, 2))) / 2
        B_mean = B.mean(dim=(1, 2))

        # Calculate gains
        rgain = G_mean / R_mean
        bgain = G_mean / B_mean

        return rgain, bgain
    

