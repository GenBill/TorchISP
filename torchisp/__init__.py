from .rggb2rgb import RGGB2RGB
from .rgb2rggb import RGB2RGGB
from .pipeline import ISP
from .inverse_pipeline import InvISP

__all__ = ['RawLoader', 'RGGB2RGB', 'RGB2RGGB', 'ISP', 'InvISP']


def __getattr__(name):
    if name == 'RawLoader':
        from .dataloader import RawLoader
        return RawLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
