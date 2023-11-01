__all__ = ["optimize"]

import pymc as pm
from pytensor.graph.basic import graph_inputs
from pytensor.tensor.variable import TensorConstant, TensorVariable


def optimize(start=None, vars=None, **kwargs):
    """A thin wrapper around ``pymc.find_MAP`` that actually does something sensible :D"""
    if vars is not None:
        if not isinstance(vars, (list, tuple)):
            vars = [vars]

        # In PyMC >= 5, model context is required to replace rvs with values
        # https://github.com/pymc-devs/pymc/pull/6281
        # https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/pymc_pytensor.html#pymc
        model = kwargs.get("model")
        model = pm.modelcontext(model)

        # find_MAP only supports passing in members of free_RVs, so let's deal
        # with that here...
        vars = model.replace_rvs_by_values(vars)
        vars = [
            v
            for v in graph_inputs(vars)
            if isinstance(v, TensorVariable)
            and not isinstance(v, TensorConstant)
        ]

    return pm.find_MAP(start=start, vars=vars, **kwargs)
