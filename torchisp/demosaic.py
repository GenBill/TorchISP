"""Optional demosaic backends for TorchISP.

The core TorchISP pipeline uses 4-channel packed Bayer tensors with channel
order matching the 2x2 Bayer tile positions: top-left, top-right, bottom-left,
bottom-right.  For the default RGGB pattern that is ``R, G1, G2, B``.

Display-oriented backends in this module accept the packed tensor after the
existing TorchISP normalization / white balance stages and return an RGB torch
``float`` tensor in the same normalized range expected by CCM and gamma stages.
Optional dependencies are imported lazily so the default PyTorch Debayer5x5 path
remains dependency-free.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import torch

SUPPORTED_BAYER_PATTERNS = ("RGGB", "GRBG", "GBRG", "BGGR")
DISPLAY_DEMOSAIC_MODES = ("amaze", "rcd", "opencv_ea")


class PackedBayerDemosaic:
    """Base adapter for display demosaic backends operating on packed Bayer.

    Args:
        bayer_pattern: Bayer tile pattern. Packed channels are interpreted as
            the four 2x2 tile positions in row-major order for that pattern.
        max_value: Integer full-scale value used for non-torch backends.
    """

    def __init__(self, bayer_pattern: str = "RGGB", max_value: int = 65535) -> None:
        self.bayer_pattern = _normalize_bayer_pattern(bayer_pattern)
        self.max_value = int(max_value)

    def __call__(self, packed_bayer: torch.Tensor) -> torch.Tensor:
        if packed_bayer.dim() != 4 or packed_bayer.shape[1] != 4:
            raise ValueError(
                "Display demosaic backends expect packed Bayer input with "
                "shape (B, 4, H, W)."
            )

        device = packed_bayer.device
        dtype = packed_bayer.dtype
        mosaics = packed_rggb_to_mosaic_uint16(packed_bayer, self.max_value)

        rgb_images = [self._demosaic_single(mosaic) for mosaic in mosaics]
        rgb = np.stack(rgb_images, axis=0).astype(np.float32) / float(self.max_value)
        rgb_tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2).to(device=device)
        return rgb_tensor.to(dtype=dtype if dtype.is_floating_point else torch.float32)

    def _demosaic_single(self, mosaic: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class AMaZEDemosaic(PackedBayerDemosaic):
    """AMaZE display demosaic adapter.

    TorchISP operates on already-loaded packed Bayer tensors rather than RAW
    containers. A tensor-native AMaZE binding is therefore required. No such
    dependency is bundled by TorchISP; this adapter imports lazily and raises a
    clear error when the optional backend is unavailable.
    """

    def __init__(self, bayer_pattern: str = "RGGB", max_value: int = 65535) -> None:
        super().__init__(bayer_pattern=bayer_pattern, max_value=max_value)
        self._backend = _import_librtprocess_backend("amaze")

    def _demosaic_single(self, mosaic: np.ndarray) -> np.ndarray:
        return self._backend(mosaic, self.bayer_pattern, self.max_value)


class RCDDemosaic(PackedBayerDemosaic):
    """RCD display demosaic adapter with lazy optional dependency handling."""

    def __init__(self, bayer_pattern: str = "RGGB", max_value: int = 65535) -> None:
        super().__init__(bayer_pattern=bayer_pattern, max_value=max_value)
        self._backend = _import_librtprocess_backend("rcd")

    def _demosaic_single(self, mosaic: np.ndarray) -> np.ndarray:
        return self._backend(mosaic, self.bayer_pattern, self.max_value)


class OpenCVEADemosaic(PackedBayerDemosaic):
    """OpenCV Edge-Aware Bayer demosaic fallback backend."""

    def __init__(self, bayer_pattern: str = "RGGB", max_value: int = 65535) -> None:
        super().__init__(bayer_pattern=bayer_pattern, max_value=max_value)
        self.cv2 = _import_cv2()
        self.code = _opencv_ea_code_map(self.cv2)[self.bayer_pattern]

    def _demosaic_single(self, mosaic: np.ndarray) -> np.ndarray:
        return self.cv2.cvtColor(mosaic, self.code)


def create_display_demosaic(
    demosaic_mode: str,
    bayer_pattern: str = "RGGB",
    max_value: int = 65535,
) -> PackedBayerDemosaic:
    """Create a display demosaic backend by mode name."""

    mode = demosaic_mode.lower()
    if mode == "amaze":
        return AMaZEDemosaic(bayer_pattern=bayer_pattern, max_value=max_value)
    if mode == "rcd":
        return RCDDemosaic(bayer_pattern=bayer_pattern, max_value=max_value)
    if mode == "opencv_ea":
        return OpenCVEADemosaic(bayer_pattern=bayer_pattern, max_value=max_value)
    raise ValueError(f"Unsupported display demosaic mode: {demosaic_mode}")


def packed_rggb_to_mosaic_uint16(packed_bayer: torch.Tensor, max_value: int = 65535) -> np.ndarray:
    """Convert packed 4-channel Bayer tensor to a uint16 2D Bayer mosaic batch.

    The packed channel order is row-major 2x2 Bayer tile order. For RGGB this is
    ``R, G1, G2, B`` and the reconstruction is::

        mosaic[:, 0::2, 0::2] = packed[:, 0]
        mosaic[:, 0::2, 1::2] = packed[:, 1]
        mosaic[:, 1::2, 0::2] = packed[:, 2]
        mosaic[:, 1::2, 1::2] = packed[:, 3]

    Input is expected to be normalized floating point data in the range used by
    the TorchISP pipeline after black-level normalization. Values are clipped to
    ``[0, 1]`` before conversion; no brightness or tone adjustment is applied.
    """

    packed_np = packed_bayer.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0).numpy()
    batch, _, height, width = packed_np.shape
    mosaic = np.empty((batch, height * 2, width * 2), dtype=np.uint16)
    scaled = np.rint(packed_np * float(max_value)).astype(np.uint16)
    mosaic[:, 0::2, 0::2] = scaled[:, 0]
    mosaic[:, 0::2, 1::2] = scaled[:, 1]
    mosaic[:, 1::2, 0::2] = scaled[:, 2]
    mosaic[:, 1::2, 1::2] = scaled[:, 3]
    return mosaic


def _normalize_bayer_pattern(bayer_pattern: str) -> str:
    pattern = bayer_pattern.upper()
    if pattern not in SUPPORTED_BAYER_PATTERNS:
        raise ValueError(
            f"Unsupported bayer_pattern: {bayer_pattern}. Supported patterns: "
            f"{', '.join(SUPPORTED_BAYER_PATTERNS)}"
        )
    return pattern


def _opencv_ea_code_map(cv2) -> Dict[str, int]:
    return {
        "RGGB": cv2.COLOR_BayerRGGB2RGB_EA,
        "GRBG": cv2.COLOR_BayerGRBG2RGB_EA,
        "GBRG": cv2.COLOR_BayerGBRG2RGB_EA,
        "BGGR": cv2.COLOR_BayerBGGR2RGB_EA,
    }


def _import_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "demosaic_mode='opencv_ea' requires OpenCV. Install it with "
            "`pip install opencv-python` or `pip install opencv-python-headless`."
        ) from exc
    return cv2


def _import_librtprocess_backend(mode: str) -> Callable[[np.ndarray, str, int], np.ndarray]:
    """Load a tensor-oriented librtprocess AMaZE/RCD binding if available.

    The upstream RawTherapee/librtprocess project is a native library and does
    not provide a universally available Python tensor API. TorchISP intentionally
    does not depend on a RAW-container reader here because inputs are packed
    tensors, not DNG/ARW/NEF files. If a project supplies a compatible Python
    binding, it can expose ``amaze_demosaic(mosaic, pattern, max_value)`` and/or
    ``rcd_demosaic(mosaic, pattern, max_value)`` on a module named
    ``librtprocess``. In practice, this means users must install or build a
    compatible Python package themselves; TorchISP does not bundle or compile
    RawTherapee/librtprocess.
    """

    try:
        import librtprocess  # type: ignore
    except ImportError as exc:
        raise ImportError(
            f"demosaic_mode='{mode}' requires a tensor-capable "
            "RawTherapee/librtprocess Python binding, which is not installed. "
            "TorchISP does not bundle this dependency; install or build a "
            "compatible Python package yourself that exposes "
            f"`{mode}_demosaic(mosaic, pattern, max_value)`. "
            "TorchISP does not silently fall back to OpenCV EA; use "
            "demosaic_mode='opencv_ea' explicitly if OpenCV quality is acceptable."
        ) from exc

    function_name = f"{mode}_demosaic"
    backend = getattr(librtprocess, function_name, None)
    if backend is None:
        raise RuntimeError(
            f"The installed librtprocess module does not expose `{function_name}`. "
            "TorchISP needs a tensor-capable demosaic function accepting "
            "(mosaic, bayer_pattern, max_value); install or build a compatible "
            "Python binding/adapter for RawTherapee/librtprocess AMaZE/RCD."
        )
    return backend
