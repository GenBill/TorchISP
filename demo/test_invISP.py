import torch
import torch.nn as nn
from torchisp.pipeline import ISP
from torchisp.inverse_pipeline import InvISP
from torchisp.dataloader import RGBLoader
from torchvision.utils import save_image

if __name__ == '__main__':
    device = 'cuda'
    rgb_path = 'rawdata/lsdir_1000.png'
    rgb_img = RGBLoader()(rgb_path).to(device)

    rggb2rgb = ISP(device=device)
    rggb2rgb.r_gain, rggb2rgb.b_gain = 1.8, 1.8

    loss_fn = nn.L1Loss() # nn.MSELoss()
    inv_isp = InvISP(loss_fn, rggb2rgb, device=device,
        lr = 1e-4, 
        nb_iter = 16000,
        eps_iter = 16 / 255,
    )

    rggb_img = inv_isp(rgb_img)
    rgb_img2 = rggb2rgb(rggb_img)

    save_image(rgb_img2, 'outputs/lsdir_1000_output.png')

