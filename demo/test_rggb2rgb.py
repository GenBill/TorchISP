
import torch
from torchvision.utils import save_image
from torchisp import RawLoader, RGGB2RGB

if __name__ == '__main__':
    
    dataloader = RawLoader(H=1520, W=2688, bl=4096)
    datapath = 'rawdata/0000.raw'

    device = 'cuda'
    rggb2rgb = RGGB2RGB(device=device)

    # rggb_img = torch.randn(1, 4, 256, 256).to(device)
    rggb_img = dataloader.get_raw16(datapath).to(device)

    rgb_img = rggb2rgb(rggb_img)

    print(rgb_img.shape)
    save_image(rgb_img, 'outputs/test_rggb2rgb.png')