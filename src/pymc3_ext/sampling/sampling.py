# -*- coding: utf-8 -*-

__all__ = ["sample"]

import numpy as np
import pymc3 as pm
from pymc3.model import all_continuous, modelcontext

from .groups import build_parameter_groups
from .quadpotential import BlockedQuadPotential
from .schedule import build_schedule
from .step_size import WindowedDualAverageAdaptation


def sample(
    *,
    draws=1000,
    tune=1000,
    model=None,
    step_kwargs=None,
    warmup_window=50,
    adapt_window=50,
    cooldown_window=100,
    initial_accept=None,
    target_accept=0.9,
    gamma=0.05,
    k=0.75,
    t0=10,
    parameter_groups=None,
    adapt_type="full",
    recompute_interval=1,
    regularization_steps=0,
    regularization_variance=1e-8,
    **kwargs,
):
    # Check that we're in a model context and that all the variables are
    # continuous
    model = modelcontext(model)
    if not all_continuous(model.vars):
        raise ValueError(
            "NUTS can only be used for models with only continuous variables."
        )

    if step_kwargs is None:
        step_kwargs = {}

    # Construct the tuning schedule using a doubling window of updates
    update_steps = build_schedule(
        tune,
        warmup_window=warmup_window,
        adapt_window=adapt_window,
        cooldown_window=cooldown_window,
    )

    # Construct the parameter groups for all the variables
    parameter_groups = build_parameter_groups(
        step_kwargs.get("vars", model.free_RVs),
        update_steps,
        parameter_groups=parameter_groups,
        adapt_type=adapt_type,
        recompute_interval=recompute_interval,
        regularization_steps=regularization_steps,
        regularization_variance=regularization_variance,
        model=model,
    )

    # Construct the potential
    potential = BlockedQuadPotential(model.ndim, parameter_groups)

    # Using this potential build a proposal step
    if "step" in kwargs:
        raise ValueError("you cannot provide a `step` argument to `sample`")
    step = pm.NUTS(potential=potential, model=model, **step_kwargs)
    potential.set_ordering(step._logp_dlogp_func._ordering)

    # Override the step size adaptation scheme using the same tuning schedule
    if "target_accept" in step_kwargs and target_accept is not None:
        raise ValueError(
            "'target_accept' cannot be given as a keyword argument and in "
            "'step_kwargs'"
        )
    target_accept = step_kwargs.pop("target_accept", target_accept)
    if initial_accept is None:
        target = target_accept
    else:
        if initial_accept > target_accept:
            raise ValueError(
                "initial_accept must be less than or equal to target_accept"
            )
        target = initial_accept + (target_accept - initial_accept) * np.sqrt(
            np.arange(len(update_steps)) / (len(update_steps) - 1)
        )
    step.step_adapt = WindowedDualAverageAdaptation(
        update_steps, step.step_size, target, gamma, k, t0
    )

    return pm.sample(draws=draws, tune=tune, model=model, step=step, **kwargs)
