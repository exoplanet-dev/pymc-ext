# -*- coding: utf-8 -*-

import numpy as np

from pymc3_ext.sampling.quadpotential import _WeightedCovariance


def get_cov(ndim):
    L = np.random.randn(ndim, ndim)
    L[np.triu_indices_from(L, 1)] = 0.0
    L[np.diag_indices_from(L)] = np.exp(L[np.diag_indices_from(L)])
    return np.dot(L, L.T)


def test_weighted_covariance(ndim=10, seed=5432):
    np.random.seed(seed)

    cov = get_cov(ndim)
    mean = np.random.randn(ndim)

    samples = np.random.multivariate_normal(mean, cov, size=100)
    mu_est0 = np.mean(samples, axis=0)
    cov_est0 = np.cov(samples, rowvar=0)

    est = _WeightedCovariance(ndim)
    for samp in samples:
        est.add_sample(samp, 1)
    mu_est = est.current_mean()
    cov_est = est.current_variance()

    assert np.allclose(mu_est, mu_est0)
    assert np.allclose(cov_est, cov_est0)

    # Make sure that the weighted estimate also works
    est2 = _WeightedCovariance(
        ndim,
        np.mean(samples[:10], axis=0),
        np.cov(samples[:10], rowvar=0, bias=True),
        10,
    )
    for samp in samples[10:]:
        est2.add_sample(samp, 1)
    mu_est2 = est2.current_mean()
    cov_est2 = est2.current_variance()

    assert np.allclose(mu_est2, mu_est0)
    assert np.allclose(cov_est2, cov_est0)
