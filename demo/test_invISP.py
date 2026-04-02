# import torch
import torch.nn as nn
from torchisp.pipeline import ISP
from torchisp.inverse_pipeline import InvISP
from torchisp.dataloader import RGBLoader
from torchisp.dataloader import save_raw4_as_bayer
from torchvision.utils import save_image

if __name__ == '__main__':
    device = 'cuda'
    rgb_path = 'rawdata/lsdir_1000.png'
    rgb_img = RGBLoader()(rgb_path).to(device)

    rggb2rgb = ISP(device=device, whitelevel=65535, blacklevel=4096)
    rggb2rgb.r_gain, rggb2rgb.b_gain = 2.0, 2.0

    loss_fn = nn.L1Loss() # nn.MSELoss()
    inv_isp = InvISP(loss_fn, rggb2rgb, device=device,
        lr = 1e-4, 
        nb_iter = 1000,
        eps_iter = 2/255,
        whitelevel=65535, blacklevel=4096
    )

    rggb_img = inv_isp(rgb_img)
    print(rggb_img.shape)

    rgb_img2 = rggb2rgb(rggb_img)
    save_image(rgb_img2, 'outputs/lsdir_1000_output.png')
    save_raw4_as_bayer(rggb_img, 'outputs/lsdir_1000_output.raw')

