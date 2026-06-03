import importlib.util

import pytest
import torch

from torchisp import ISP
from torchisp.demosaic import packed_rggb_to_mosaic_uint16


class IdentityCCM(torch.nn.Module):
    def forward(self, rgb):
        return rgb


class FixedWB(torch.nn.Module):
    def forward(self, rggb, r_gain=None, b_gain=None):
        return rggb


class MeanDebayer(torch.nn.Module):
    def forward(self, bayer):
        rgb = bayer.repeat(1, 3, 1, 1)
        return rgb


def _raw_input(batch=1, height=8, width=8):
    return torch.rand(batch, 4, height, width) * (65535 - 4096) + 4096


def test_isp_default_matches_explicit_debayer5x5():
    torch.manual_seed(0)
    default_isp = ISP(device="cpu")
    explicit_isp = ISP(device="cpu", demosaic_mode="debayer5x5")
    explicit_isp.load_state_dict(default_isp.state_dict())

    rggb = _raw_input()

    with torch.no_grad():
        default_rgb = default_isp(rggb)
        explicit_rgb = explicit_isp(rggb)

    assert default_rgb.shape == (1, 3, 16, 16)
    assert torch.allclose(default_rgb, explicit_rgb)


def test_explicit_debayer_argument_is_used_for_debayer5x5():
    isp = ISP(
        device="cpu",
        Debayer=MeanDebayer(),
        wbcFunc=FixedWB(),
        ccm_matrix=torch.eye(3),
        gamma=1.0,
        demosaic_mode="debayer5x5",
    )
    isp.ccmFunc = IdentityCCM()
    rggb = torch.ones(1, 4, 2, 2)

    with torch.no_grad():
        rgb = isp(rggb, apply_normalize=False)

    assert rgb.shape == (1, 3, 4, 4)
    assert torch.allclose(rgb, torch.ones_like(rgb))


def test_invalid_demosaic_mode_raises_clear_error():
    with pytest.raises(ValueError, match="Unsupported demosaic_mode"):
        ISP(device="cpu", demosaic_mode="not_a_backend")


def test_default_import_does_not_require_optional_display_dependencies():
    assert importlib.util.find_spec("torchisp.demosaic") is not None
    isp = ISP(device="cpu", demosaic_mode="debayer5x5")
    assert isp.demosaic_mode == "debayer5x5"


def test_amaze_unavailable_raises_without_opencv_fallback():
    if importlib.util.find_spec("librtprocess") is not None:
        pytest.skip("librtprocess is installed in this environment")

    with pytest.raises(ImportError, match="does not silently fall back"):
        ISP(device="cpu", demosaic_mode="amaze")


def test_packed_rggb_to_mosaic_uint16_channel_order():
    packed = torch.tensor([[[[1.0]], [[0.5]], [[0.25]], [[0.0]]]])
    mosaic = packed_rggb_to_mosaic_uint16(packed)

    assert mosaic.shape == (1, 2, 2)
    assert mosaic[0, 0, 0] == 65535
    assert mosaic[0, 0, 1] == round(0.5 * 65535)
    assert mosaic[0, 1, 0] == round(0.25 * 65535)
    assert mosaic[0, 1, 1] == 0


@pytest.mark.skipif(importlib.util.find_spec("cv2") is None, reason="OpenCV is not installed")
def test_opencv_ea_output_shape_dtype_and_range():
    isp = ISP(
        device="cpu",
        demosaic_mode="opencv_ea",
        wbcFunc=FixedWB(),
        ccm_matrix=torch.eye(3),
        gamma=1.0,
    )
    isp.ccmFunc = IdentityCCM()
    rggb = torch.rand(2, 4, 6, 5)

    with torch.no_grad():
        rgb = isp(rggb, apply_normalize=False)

    assert rgb.shape == (2, 3, 12, 10)
    assert rgb.dtype == rggb.dtype
    assert torch.all(rgb >= 0.0)
    assert torch.all(rgb <= 1.0)
