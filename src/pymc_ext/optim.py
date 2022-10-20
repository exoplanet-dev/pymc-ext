__all__ = ["optimize"]

import pymc as pm
from aesara.graph.basic import graph_inputs
from aesara.tensor.var import TensorConstant, TensorVariable


def optimize(start=None, vars=None, **kwargs):
    """A thin wrapper around ``pymc.find_MAP`` that actually does something sensible :D"""
    if vars is not None:
        if not isinstance(vars, (list, tuple)):
            vars = [vars]

        # find_MAP only supports passing in members of free_RVs, so let's deal
        # with that here...
        vars = pm.aesaraf.rvs_to_value_vars(vars)
        vars = [
            v
            for v in graph_inputs(vars)
            if isinstance(v, TensorVariable)
            and not isinstance(v, TensorConstant)
        ]

    return pm.find_MAP(start=start, vars=vars, **kwargs)
