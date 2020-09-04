# -*- coding: utf-8 -*-

__all__ = ["optimize"]

import sys

import numpy as np
import pymc3 as pm
import theano
from fastprogress.fastprogress import progress_bar
from pymc3.blocking import ArrayOrdering, DictToArrayBijection
from pymc3.model import Point
from pymc3.theanof import inputvars
from pymc3.util import (
    get_default_varnames,
    get_untransformed_name,
    is_transformed_name,
    update_start_vals,
)
from scipy.optimize import minimize

from .utils import (
    get_args_for_theano_function,
    get_theano_function_for_var,
    logger,
)


def start_optimizer(maxeval, vars, verbose=True, progress=True, **kwargs):
    progressbar = None
    if verbose:
        names = [
            get_untransformed_name(v.name)
            if is_transformed_name(v.name)
            else v.name
            for v in vars
        ]
        sys.stderr.write(
            "optimizing logp for variables: [{0}]\n".format(", ".join(names))
        )

        if progress:
            progressbar = progress_bar(range(maxeval), total=maxeval)
            progressbar.update(0)

    return progressbar


def get_point(wrapper, x):
    vars = get_default_varnames(wrapper.model.unobserved_RVs, True)
    return {
        var.name: value
        for var, value in zip(
            vars, wrapper.model.fastfn(vars)(wrapper.bij.rmap(x))
        )
    }


def optimize(
    start=None,
    vars=None,
    return_info=False,
    verbose=True,
    progress=True,
    maxeval=5000,
    model=None,
    **kwargs
):
    """Maximize the log prob of a PyMC3 model using scipy

    All extra arguments are passed directly to the ``scipy.optimize.minimize``
    function.

    Args:
        start: The PyMC3 coordinate dictionary of the starting position
        vars: The variables to optimize
        model: The PyMC3 model
        return_info: Return both the coordinate dictionary and the result of
            ``scipy.optimize.minimize``
        verbose: Print the success flag and log probability to the screen
        progress_bar: A ``tqdm`` progress bar instance. Set to ``'auto'``
            to use ``tqdm.auto.tqdm()``, ``'tqdm'`` to use ``tqdm.tqdm()``.
            Set to ``None`` to disable (default).

    """
    wrapper = ModelWrapper(start=start, vars=vars, model=model)
    progressbar = start_optimizer(
        maxeval, wrapper.vars, verbose=verbose, progress=progress
    )

    # Count the number of function calls
    neval = 0

    # This returns the objective function and its derivatives
    def objective(vec):
        nonlocal neval
        neval += 1
        nll, grad = wrapper(vec)
        if progressbar:
            progressbar.comment = "logp = {0:.3e}".format(-nll)
            progressbar.update_bar(neval)

        if neval > maxeval:
            raise StopIteration

        return nll, grad

    # Optimize using scipy.optimize
    x0 = wrapper.bij.map(wrapper.start)
    initial = objective(x0)[0]
    kwargs["jac"] = True

    try:
        info = minimize(objective, x0, **kwargs)
    except (KeyboardInterrupt, StopIteration):
        info = None
    finally:
        if progressbar:
            progressbar.total = neval
            progressbar.update(neval)
            print()

    # Only accept the output if it is better than it was
    x = info.x if info and np.isfinite(info.fun) and info.fun < initial else x0

    # Coerce the output into the right format
    point = get_point(wrapper, x)

    if verbose:
        sys.stderr.write("message: {0}\n".format(info.message))
        sys.stderr.write("logp: {0} -> {1}\n".format(-initial, -info.fun))
        if not np.isfinite(info.fun):
            logger.warning("final logp not finite, returning initial point")
            logger.warning(
                "this suggests that something is wrong with the model"
            )
            logger.debug("{0}".format(info))

    if return_info:
        return point, info
    return point


def optimize_iterator(
    stepper,
    maxiter=1000,
    start=None,
    vars=None,
    model=None,
):
    """Maximize the log prob of a PyMC3 model using a custom stepper function

    Args:
        stepper: An optimizer object
        maxiter: The maximum number of steps to run
        start: The PyMC3 coordinate dictionary of the starting position
        vars: The variables to optimize
        model: The PyMC3 model

    """
    wrapper = ModelWrapper(start=start, vars=vars, model=model)
    x = wrapper.bij.map(wrapper.start)

    n = 0
    stepper.reset()
    while True:
        x, nll = stepper.step(wrapper, x)
        yield nll, get_point(wrapper, x)
        n += 1
        if maxiter is not None and n >= maxiter:
            break


def allinmodel(vars, model):
    notin = [v for v in vars if v not in model.vars]
    if notin:
        raise ValueError("Some variables not in the model: " + str(notin))


class ModelWrapper:
    def __init__(self, start=None, vars=None, model=None):
        model = self.model = pm.modelcontext(model)

        # Work out the full starting coordinates
        if start is None:
            start = model.test_point
        else:
            update_start_vals(start, model.test_point, model)
        self.start = start

        # Fit all the parameters by default
        if vars is None:
            vars = model.cont_vars
        vars = self.vars = inputvars(vars)
        allinmodel(vars, model)

        # Work out the relevant bijection map
        start = Point(start, model=model)
        self.bij = DictToArrayBijection(ArrayOrdering(vars), start)

        # Pre-compile the theano model and gradient
        nlp = -model.logpt
        grad = theano.grad(nlp, vars, disconnected_inputs="ignore")
        self.func = get_theano_function_for_var([nlp] + grad, model=model)

    def __call__(self, vec):
        try:
            res = self.func(
                *get_args_for_theano_function(
                    self.bij.rmap(vec), model=self.model
                )
            )
        except Exception:
            import traceback

            print("array:", vec)
            print("point:", self.bij.rmap(vec))
            traceback.print_exc()
            raise

        d = dict(zip((v.name for v in self.vars), res[1:]))
        g = self.bij.map(d)
        return res[0], g


class Adam:
    """https://github.com/pytorch/pytorch/blob/master/torch/optim/adam.py"""

    def __init__(
        self,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1])
            )
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.reset()

    def reset(self):
        self.state = {"step": 0}

    def step(self, loss_and_grad_func, p):
        loss, grad = loss_and_grad_func(p)

        state = self.state
        if state["step"] == 0:
            # Exponential moving average of gradient values
            state["exp_avg"] = np.zeros_like(p)
            # Exponential moving average of squared gradient values
            state["exp_avg_sq"] = np.zeros_like(p)
            if self.amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state["max_exp_avg_sq"] = np.zeros_like(p)

        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        if self.amsgrad:
            max_exp_avg_sq = state["max_exp_avg_sq"]
        beta1, beta2 = self.betas

        state["step"] += 1
        bias_correction1 = 1 - beta1 ** state["step"]
        bias_correction2 = 1 - beta2 ** state["step"]

        if self.weight_decay != 0:
            grad[:] += self.weight_decay * p

        # Decay the first and second moment running average coefficient
        exp_avg[:] *= beta1
        exp_avg[:] += (1 - beta1) * grad
        exp_avg_sq[:] *= beta2
        exp_avg_sq[:] += (1 - beta2) * grad ** 2
        if self.amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            max_exp_avg_sq[:] = np.maximum(max_exp_avg_sq, exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = (
                np.sqrt(max_exp_avg_sq) / np.sqrt(bias_correction2) + self.eps
            )
        else:
            denom = np.sqrt(exp_avg_sq) / np.sqrt(bias_correction2) + self.eps

        step_size = self.lr / bias_correction1

        return p - step_size * exp_avg / denom, loss
