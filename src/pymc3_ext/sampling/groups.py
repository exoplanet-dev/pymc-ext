# -*- coding: utf-8 -*-

__all__ = ["ParameterGroup"]

import numpy as np
import pymc3 as pm
from pymc3.theanof import inputvars

from .quadpotential import WindowedDiagAdapt, WindowedFullAdapt


def allinmodel(vars, model):
    notin = [v for v in vars if v not in model.free_RVs]
    if notin:
        raise ValueError("Some variables not in the model: " + str(notin))


class ParameterGroup:
    allowed_adapt_types = ("full", "diag")

    def __init__(self, variables, adapt_type="full", model=None):
        if adapt_type not in self.allowed_adapt_types:
            raise ValueError(
                "`adapt_type` must be one of {0}".format(
                    self.allowed_adapt_types
                )
            )
        model = pm.modelcontext(model)
        self.variables = inputvars(variables)
        allinmodel(self.variables, model)
        self.adapt_type = adapt_type

    def compute_indices(self, ordering, update_steps, **kwargs):
        inds = np.arange(ordering.size)
        these_inds = []
        for v in self.variables:
            these_inds.append(inds[ordering[v.name].slc])
        self.indices = np.sort(np.concatenate(these_inds))

        self.potential = dict(full=WindowedFullAdapt, diag=WindowedDiagAdapt)[
            self.adapt_type
        ](len(self.indices), update_steps, **kwargs)
