from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pyDowker")
except PackageNotFoundError:
    __version__ = "unknown"

from .DowkerComplex import DowkerComplex

__all__ = ["DowkerComplex", "__version__"]
