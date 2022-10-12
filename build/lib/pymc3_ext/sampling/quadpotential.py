# -*- coding: utf-8 -*-

__all__ = ["BlockedQuadPotential", "WindowedDiagAdapt", "WindowedFullAdapt"]

import numpy as np
from pymc3.step_methods.hmc.quadpotential import (
    QuadPotential,
    _WeightedVariance,
)
from scipy.linalg import LinAlgError, cholesky, solve_triangular

from .estimator import _WeightedCovariance


class BlockedQuadPotential(QuadPotential):
    def __init__(self, n, groups, dtype="float64"):
        self.dtype = dtype
        self.n = int(n)
        self.groups = groups
        self.ordering = None
        self.vmap = None

    def set_ordering(self, ordering):
        self.ordering = ordering
        self.vmap = []
        inds = np.arange(self.n)
        for group in self.groups:
            self.vmap.append(
                np.concatenate(
                    [inds[self.ordering[v.name].slc] for v in group.variables]
                )
            )

    def reset(self):
        for group in self.groups:
            group.potential.reset()

    def velocity(self, x, out=None):
        if out is None:
            out = np.zeros_like(x)
        for inds, group in zip(self.vmap, self.groups):
            out[inds] = group.potential.velocity(x[inds])
        return out

    def energy(self, x, velocity=None):
        if velocity is None:
            velocity = self.velocity(x)
        return 0.5 * np.dot(x, velocity)

    def velocity_energy(self, x, v_out):
        self.velocity(x, out=v_out)
        return self.energy(x, v_out)

    def random(self):
        out = np.empty(self.n)
        for inds, group in zip(self.vmap, self.groups):
            out[inds] = group.potential.random()
        return out

    def update(self, sample, grad, tune):
        if not tune:
            return

        for inds, group in zip(self.vmap, self.groups):
            group.potential.update(sample[inds], grad[inds], tune)

    def raise_ok(self, vmap):
        for group in self.groups:
            group.potential.raise_ok(vmap)


class WindowedDiagAdapt(QuadPotential):
    def __init__(
        self,
        ndim,
        update_steps=None,
        recompute_interval=1,
        regularization_steps=0,
        regularization_variance=1e-8,
        dtype="float64",
    ):
        self.dtype = dtype
        self._ndim = int(ndim)

        if update_steps is not None:
            self._update_steps = np.atleast_1d(update_steps).astype(int)
        else:
            self._update_steps = np.array([], dtype=int)
        self._recompute_interval = int(recompute_interval)

        self._regularization_steps = int(regularization_steps)
        self._regularization_variance = float(regularization_variance)

        self.reset()

    def reset(self):
        self._n_samples = 0

        self.new_variance()
        self.update_factors()

        self._foreground = self.new_estimator()
        self._background = self.new_estimator()

    def update(self, sample, grad, tune):
        if not tune:
            return

        self._n_samples += 1

        # If we're in warmup or cooldown, we shouldn't update the variance
        if (
            self._n_samples <= self._update_steps[0]
            or self._n_samples > self._update_steps[-1]
        ):
            return

        # Add the sample to the estimators
        self._foreground.add_sample(sample, weight=1)
        self._background.add_sample(sample, weight=1)

        # During the first slow window, never update the variance estimate
        if self._n_samples < self._update_steps[1]:
            return

        # If this is one of the update steps, update the estimators
        if self._n_samples in self._update_steps:
            self._foreground = self._background
            self._background = self.new_estimator()
            self.update_var()

        # Update the variance every `recompute_interval` steps
        elif (
            self._recompute_interval
            and self._n_samples % self._recompute_interval == 0
        ):
            self.update_var()

    def set_var(self, var):
        self._var = var
        self.update_factors()

    def update_var(self):
        self._foreground.current_variance(out=self._var)

        if self._regularization_steps > 0:
            N = self._foreground.n_samples
            n = self._regularization_steps
            self._var *= N / (N + n)
            self._var[self._diag_inds] += (
                self._regularization_variance * n / (N + n)
            )

        self.update_factors()

    def energy(self, x, velocity=None):
        if velocity is None:
            velocity = self.velocity(x)
        return 0.5 * x.dot(velocity)

    def velocity_energy(self, x, v_out):
        self.velocity(x, out=v_out)
        return 0.5 * np.dot(x, v_out)

    #
    # The following methods should be overloaded by subclasses
    #

    def new_estimator(self):
        return _WeightedVariance(self._ndim, dtype=self.dtype)

    def new_variance(self):
        self._var = np.ones(self._ndim, dtype=self.dtype)
        self._diag_inds = np.arange(self._ndim)

    def update_factors(self):
        self._inv_sd = 1.0 / np.sqrt(self._var)

    def velocity(self, x, out=None):
        return np.multiply(self._var, x, out=out)

    def random(self):
        vals = np.random.normal(size=self._ndim).astype(self.dtype)
        return self._inv_sd * vals

    def raise_ok(self, vmap):
        if np.any(~np.isfinite(self._inv_sd)):
            raise ValueError("non-finite inverse variances found")


class WindowedFullAdapt(WindowedDiagAdapt):
    def new_estimator(self):
        return _WeightedCovariance(self._ndim, dtype=self.dtype)

    def new_variance(self):
        self._var = np.eye(self._ndim, dtype=self.dtype)
        self._diag_inds = np.diag_indices(self._ndim)

    def update_factors(self):
        try:
            self._chol = cholesky(self._var, lower=True)
        except (LinAlgError, ValueError) as error:
            self._chol_error = error
        else:
            self._chol_error = None

    def velocity(self, x, out=None):
        return np.dot(self._var, x, out=out)

    def random(self):
        vals = np.random.normal(size=self._ndim).astype(self.dtype)
        return solve_triangular(self._chol.T, vals, overwrite_b=True)

    def raise_ok(self, vmap):
        if self._chol_error is not None:
            raise ValueError("{0}".format(self._chol_error))
