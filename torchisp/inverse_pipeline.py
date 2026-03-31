import torch
import torch.nn as nn
from torchisp.attacker.pgd import TargetLinfPGD
from torchisp.attacker.adampgd import AdamPGD
from torchisp import RGB2RGGB

class InvISP(nn.Module):
    """
    Inverse ISP Pipeline

    This class takes an RGB image and reconstructs the original RGGB RAW data using an adversarial optimization approach.
    The optimization starts from a initialized RGGB input, iteratively adjusted using the AdamPGD method.

    Arguments:
    - loss_fn (callable): Loss function used to calculate the adversarial loss.
    - rggb2rgb (callable): Function to convert RGGB to RGB.
    - lr (float): Learning rate for the AdamPGD optimizer (default: 1e-4).
    - nb_iter (int): Number of iterations for the AdamPGD optimization (default: 1000).
    - eps_iter (float): Maximum perturbation step for each iteration (default: 4/255).
    - device (str): Device to run the computations on (default: 'cuda').
    """
    def __init__(self, loss_fn, rggb2rgb, lr=1e-4, nb_iter=1000, eps_iter=4/255, 
        device='cuda', whitelevel=65535, blacklevel=4096
    ) -> None:
        super(InvISP, self).__init__()
        self.device = device
        self.whitelevel = whitelevel
        self.blacklevel = blacklevel
        self.nb_iter = nb_iter
        self.lr = lr
        self.eps_iter = eps_iter
        self.loss_fn = loss_fn
        self.rggb2rgb = rggb2rgb
        
        # AdamPGD optimizes in [0,1] RGGB; ISP must not run normalize() on that tensor again.
        def predict_no_norm(x):
            return self.rggb2rgb(x, apply_normalize=False)
        self.attacker = AdamPGD(predict=predict_no_norm, loss_fn=self.loss_fn, lr=self.lr, nb_iter=self.nb_iter, eps_iter=self.eps_iter)
        # self.attacker = TargetLinfPGD(predict=self.rggb2rgb, loss_fn=self.loss_fn, nb_iter=self.nb_iter, eps_iter=self.eps_iter)

    def normalize(self, rggb):
        return (rggb - self.blacklevel) / (self.whitelevel - self.blacklevel)
    
    def denormalize(self, rggb):
        return rggb * (self.whitelevel - self.blacklevel) + self.blacklevel
    
    def forward(self, rgb):
        """
        Reconstruct the RGGB RAW image from an RGB image using the inverse ISP.

        Arguments:
        - rgb (torch.Tensor): RGB image tensor, assumed to be in range [0, 1].

        Returns:
        - torch.Tensor: Reconstructed RGGB RAW tensor.
        """
        # # Initialize RGGB with zeros
        # rggb_init = torch.zeros((rgb.shape[0], 4, rgb.shape[2] // 2, rgb.shape[3] // 2), device=self.device)
        # Initialize RGGB with rgb_unprocess
        rggb_init = RGB2RGGB(device=self.device)(rgb)
        
        # Use AdamPGD to optimize from the zero-initialization
        rggb_optimized = self.attacker.perturb(rggb_init, rgb)

        rggb_optimized = self.denormalize(rggb_optimized)
        
        return rggb_optimized

# Example usage
if __name__ == "__main__":
    loss_fn = nn.MSELoss()

    def rggb2rgb(x, apply_normalize=False):  # signature matches ISP for InvISP inner predict
        return x
    
    lr = 1e-4
    nb_iter = 1000
    eps_iter = 2/255
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = InvISP(loss_fn=loss_fn, rggb2rgb=rggb2rgb, lr=lr, nb_iter=nb_iter, eps_iter=eps_iter, device=device, whitelevel=65535, blacklevel=4096)
    
    # Placeholder RGB image tensor with batch size of 1
    rgb_image = torch.rand((1, 3, 256, 256), device=device)
    
    rggb_output = model(rgb_image)
    print(rggb_output.shape)  # Expected output shape: (1, 4, 128, 128) for RGGB output
