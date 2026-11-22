"""Find project code that duplicates functions or constants its dependencies export."""

from importlib.metadata import PackageNotFoundError, version

from .config import Config, load_config
from .extract import Const, Func
from .gate import Result, check, report
from .match import Match, find, find_constants

try:
    __version__ = version("pysayme")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"

__all__ = [
    "Config",
    "Const",
    "Func",
    "Match",
    "Result",
    "check",
    "find",
    "find_constants",
    "load_config",
    "report",
]
