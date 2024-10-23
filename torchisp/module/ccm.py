import torch
import numpy as np

class SimpleCCM(torch.nn.Module):

    def __init__(self, device='cpu', ccm_matrix=None):
        super().__init__()
        
        self.device = device
        
        if ccm_matrix is None:
            self.ccm = torch.tensor(
                [[1.906, -0.422, -0.477], 
                [-0.516, 2.0, -0.484], 
                [-0.063, -0.852, 1.914]], device=self.device)
        elif isinstance(ccm_matrix, np.ndarray):
            # Convert numpy array to torch tensor
            self.ccm = torch.tensor(ccm_matrix, device=self.device)
        elif isinstance(ccm_matrix, torch.Tensor):
            self.ccm = ccm_matrix.to(self.device)
        else:
            raise ValueError("ccm_matrix must be either None, a numpy array, or a torch tensor.")

    def forward(self, inp_rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inp_rgb: (N, 3, H, W) tensor
        Returns:
            rgb: (N, 3, H, W) tensor
        """
        assert inp_rgb.shape[1] == 3 and inp_rgb.dim() == 4, "Input should have shape (N, 3, H, W)"

        # Permute to (N, H, W, 3) for matrix multiplication
        rgb = inp_rgb.permute(0, 2, 3, 1)
        ccm_tensor = self.ccm.transpose(0, 1).to(inp_rgb.device)

        # Apply CCM
        rgb = torch.matmul(rgb, ccm_tensor)

        # Permute back to (N, 3, H, W) and clamp to specified range
        rgb = rgb.permute(0, 3, 1, 2)

        return rgb
