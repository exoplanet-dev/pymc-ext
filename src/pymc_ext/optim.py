__all__ = ["optimize"]

import pymc as pm


def optimize(start=None, vars=None, **kwargs):
    """A thin wrapper around ``pymc.find_MAP`` that can handle deterministics"""
    if vars is not None:
        if not isinstance(vars, (list, tuple)):
            vars = [vars]
        vars = pm.aesaraf.rvs_to_value_vars(vars)
    return pm.find_MAP(start=start, vars=vars, **kwargs)
