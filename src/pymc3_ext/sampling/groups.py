# -*- coding: utf-8 -*-

__all__ = ["ParameterGroup"]

import pymc3 as pm
from pymc3.blocking import ArrayOrdering
from pymc3.theanof import inputvars

from .quadpotential import WindowedDiagAdapt, WindowedFullAdapt


def build_parameter_groups(
    variables,
    update_steps,
    *,
    parameter_groups=None,
    adapt_type="full",
    recompute_interval=1,
    regularization_steps=0,
    regularization_variance=1e-8,
    model=None,
):
    # Check the parameter groups
    if parameter_groups is None:
        parameter_groups = []
    remaining = set(variables)
    for i, group in enumerate(parameter_groups):
        try:
            this_group = set(group.variables)
        except AttributeError:
            this_group = set(group)
            parameter_groups[i] = ParameterGroup(list(this_group), model=model)
        if len(this_group & remaining) != len(this_group):
            raise ValueError("parameters can only be included in one group")
        remaining -= this_group
    if len(remaining):
        parameter_groups.append(
            ParameterGroup(list(remaining), adapt_type=adapt_type, model=model)
        )

    for group in parameter_groups:
        group.setup_potential(
            update_steps,
            recompute_interval=recompute_interval,
            regularization_steps=regularization_steps,
            regularization_variance=regularization_variance,
        )

    return parameter_groups


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
        self.size = ArrayOrdering(variables).size

    def setup_potential(self, update_steps, **kwargs):
        self.potential = dict(full=WindowedFullAdapt, diag=WindowedDiagAdapt)[
            self.adapt_type
        ](self.size, update_steps, **kwargs)
