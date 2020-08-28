# -*- coding: utf-8 -*-

__all__ = ["QuadPotentialDenseAdapt", "get_dense_nuts_step"]

import numpy as np
import pymc3 as pm
from pymc3.model import all_continuous, modelcontext
from pymc3.step_methods.hmc.quadpotential import QuadPotential
from scipy.linalg import LinAlgError, cholesky, solve_triangular

from ..utils import logger
from .estimator import _WeightedCovariance


class QuadPotentialDenseAdapt(QuadPotential):
    """Adapt a dense mass matrix from the sample covariances."""

    def __init__(
        self,
        n,
        initial_mean=None,
        initial_cov=None,
        initial_weight=0,
        adaptation_window=101,
        doubling=True,
        update_steps=None,
        dtype="float64",
    ):
        if initial_mean is None:
            initial_mean = np.zeros(n, dtype=dtype)
        if initial_cov is None:
            initial_cov = np.eye(n, dtype=dtype)
            initial_weight = 1

        if initial_cov is not None and initial_cov.ndim != 2:
            raise ValueError("Initial covariance must be two-dimensional.")
        if initial_mean is not None and initial_mean.ndim != 1:
            raise ValueError("Initial mean must be one-dimensional.")
        if initial_cov is not None and initial_cov.shape != (n, n):
            raise ValueError(
                "Wrong shape for initial_cov: expected %s got %s"
                % (n, initial_cov.shape)
            )
        if len(initial_mean) != n:
            raise ValueError(
                "Wrong shape for initial_mean: expected %s got %s"
                % (n, len(initial_mean))
            )

        self.dtype = dtype
        self._n = n
        self._cov = np.array(initial_cov, dtype=self.dtype, copy=True)
        self._chol = cholesky(self._cov, lower=True)
        self._chol_error = None
        self._foreground_cov = _WeightedCovariance(
            self._n, initial_mean, initial_cov, initial_weight, self.dtype
        )
        self._background_cov = _WeightedCovariance(self._n, dtype=self.dtype)
        self._n_samples = 0

        # For backwards compatibility
        self._doubling = doubling
        self._adaptation_window = int(adaptation_window)
        self._previous_update = 0

        # New interface
        if update_steps is None:
            self._update_steps = None
        else:
            self._update_steps = np.atleast_1d(update_steps).astype(int)

    def velocity(self, x, out=None):
        return np.dot(self._cov, x, out=out)

    def energy(self, x, velocity=None):
        if velocity is None:
            velocity = self.velocity(x)
        return 0.5 * np.dot(x, velocity)

    def velocity_energy(self, x, v_out):
        self.velocity(x, out=v_out)
        return self.energy(x, v_out)

    def random(self):
        vals = np.random.normal(size=self._n).astype(self.dtype)
        return solve_triangular(self._chol.T, vals, overwrite_b=True)

    def _update_from_weightvar(self, weightvar):
        weightvar.current_covariance(out=self._cov)
        N = weightvar.n_samples
        n = 10
        self._cov *= N / (N + n)
        self._cov[np.diag_indices_from(self._cov)] += 1e-8 * n / (N + n)
        try:
            self._chol = cholesky(self._cov, lower=True)
        except (LinAlgError, ValueError) as error:
            self._chol_error = error

    def update(self, sample, grad, tune):
        if not tune:
            return

        self._foreground_cov.add_sample(sample, weight=1)
        self._background_cov.add_sample(sample, weight=1)
        self._update_from_weightvar(self._foreground_cov)

        # Support the two methods for updating the mass matrix
        delta = self._n_samples - self._previous_update
        do_update = (
            self._update_steps is not None
            and self._n_samples in self._update_steps
        ) or (self._update_steps is None and delta >= self._adaptation_window)
        if do_update:
            self._foreground_cov = self._background_cov
            self._background_cov = _WeightedCovariance(
                self._n, dtype=self.dtype
            )

            if self._update_steps is None:
                self._previous_update = self._n_samples
                if self._doubling:
                    self._adaptation_window *= 2

        self._n_samples += 1

    def raise_ok(self, vmap):
        if self._chol_error is not None:
            raise ValueError("{0}".format(self._chol_error))


def get_dense_nuts_step(
    start=None,
    adaptation_window=101,
    doubling=True,
    initial_weight=10,
    use_hessian=False,
    use_hessian_diag=False,
    hessian_regularization=1e-8,
    model=None,
    **kwargs,
):
    """Get a NUTS step function with a dense mass matrix

    The entries in the mass matrix will be tuned based on the sample
    covariances during tuning. All extra arguments are passed directly to
    ``pymc3.NUTS``.

    Args:
        start (dict, optional): A starting point in parameter space. If not
            provided, the model's ``test_point`` is used.
        adaptation_window (int, optional): The (initial) size of the window
            used for sample covariance estimation.
        doubling (bool, optional): If ``True`` (default) the adaptation window
            is doubled each time the matrix is updated.

    """
    model = modelcontext(model)

    if not all_continuous(model.vars):
        raise ValueError(
            "NUTS can only be used for models with only "
            "continuous variables."
        )

    if start is None:
        start = model.test_point
    mean = model.dict_to_array(start)

    if use_hessian or use_hessian_diag:
        try:
            import numdifftools as nd
        except ImportError:
            raise ImportError(
                "The 'numdifftools' package is required for Hessian "
                "computations"
            )

        logger.info("Numerically estimating Hessian matrix")
        if use_hessian_diag:
            hess = nd.Hessdiag(model.logp_array)(mean)
            var = np.diag(-1.0 / hess)
        else:
            hess = nd.Hessian(model.logp_array)(mean)
            var = -np.linalg.inv(hess)

        factor = 1
        success = False
        while not success:
            var[np.diag_indices_from(var)] += factor * hessian_regularization

            try:
                np.linalg.cholesky(var)
            except np.linalg.LinAlgError:
                factor *= 2
            else:
                success = True

    else:
        var = np.eye(len(mean))

    potential = QuadPotentialDenseAdapt(
        model.ndim,
        initial_mean=mean,
        initial_cov=var,
        initial_weight=initial_weight,
        adaptation_window=adaptation_window,
        doubling=doubling,
    )

    return pm.NUTS(potential=potential, model=model, **kwargs)
