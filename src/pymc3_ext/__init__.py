# -*- coding: utf-8 -*-

__all__ = [
    "distributions",
    "optim",
    "sampling",
    "optimize",
    "__version__",
    "sample",
    "eval_in_model",
    "get_samples_from_trace",
    "get_args_for_theano_function",
    "get_theano_function_for_var",
]

from . import distributions, optim, sampling
from .optim import optimize
from .pymc3_ext_version import __version__
from .sampling import sample
from .utils import (
    eval_in_model,
    get_args_for_theano_function,
    get_samples_from_trace,
    get_theano_function_for_var,
)

__uri__ = "https://docs.exoplanet.codes"
__author__ = "Daniel Foreman-Mackey"
__email__ = "foreman.mackey@gmail.com"
__license__ = "MIT"
__description__ = "Useful(?) extensions to PyMC3"
