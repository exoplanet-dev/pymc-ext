# -*- coding: utf-8 -*-

__all__ = [
    "optim",
    "sampling",
    "optimize",
    "__version__",
    "sample",
]

from . import optim, sampling
from .optim import optimize
from .pymc3_ext_version import __version__
from .sampling import sample
from .distributions import *  # noqa

__uri__ = "https://github.com/exoplanet-dev/pymc3-ext"
__author__ = "Daniel Foreman-Mackey"
__email__ = "foreman.mackey@gmail.com"
__license__ = "MIT"
__description__ = 'PyMC3 Extras extracted from the "exoplanet" library'
