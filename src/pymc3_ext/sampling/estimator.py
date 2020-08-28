# -*- coding: utf-8 -*-

__all__ = ["_WeightedCovariance"]

import numpy as np


class _WeightedCovariance:
    """Online algorithm for computing mean and covariance."""

    def __init__(
        self,
        nelem,
        initial_mean=None,
        initial_covariance=None,
        initial_weight=0,
        dtype="float64",
    ):
        self._dtype = dtype
        self.n_samples = float(initial_weight)
        if initial_mean is None:
            self.mean = np.zeros(nelem, dtype=dtype)
        else:
            self.mean = np.array(initial_mean, dtype=dtype, copy=True)
        if initial_covariance is None:
            self.raw_cov = np.eye(nelem, dtype=dtype)
        else:
            self.raw_cov = np.array(initial_covariance, dtype=dtype, copy=True)

        self.raw_cov[:] *= self.n_samples

        if self.raw_cov.shape != (nelem, nelem):
            raise ValueError("Invalid shape for initial covariance.")
        if self.mean.shape != (nelem,):
            raise ValueError("Invalid shape for initial mean.")

    def add_sample(self, x, weight):
        x = np.asarray(x)
        self.n_samples += 1
        old_diff = x - self.mean
        self.mean[:] += old_diff / self.n_samples
        new_diff = x - self.mean
        self.raw_cov[:] += weight * new_diff[:, None] * old_diff[None, :]

    def current_variance(self, out=None):
        if self.n_samples == 0:
            raise ValueError("Can not compute covariance without samples.")
        if out is not None:
            return np.divide(self.raw_cov, self.n_samples - 1, out=out)
        else:
            return (self.raw_cov / (self.n_samples - 1)).astype(self._dtype)

    def current_mean(self):
        return np.array(self.mean, dtype=self._dtype)
