
import torch
from torchisp import RGGB2RGB

if __name__ == '__main__':
    
    device = 'cpu'
    rggb2rgb = RGGB2RGB(device=device)

    rggb_img = torch.randn(1, 4, 256, 256).to(device)

    rgb_img = rggb2rgb(rggb_img)

    print(rgb_img.shape)
