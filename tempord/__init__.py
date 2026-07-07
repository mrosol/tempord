from importlib.metadata import version, PackageNotFoundError

from .tempord import tempord, get_causal_vector
from .plots import make_plot

__all__ = ["tempord", "get_causal_vector", "make_plot"]

try:
    __version__ = version("tempord")
except PackageNotFoundError:
    __version__ = "0.0.0"
