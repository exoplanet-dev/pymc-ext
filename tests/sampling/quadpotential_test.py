# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pymc3_ext.sampling.quadpotential import (
    WindowedDiagAdapt,
    WindowedFullAdapt,
)


def get_cov(ndim):
    L = np.random.randn(ndim, ndim)
    L[np.triu_indices_from(L, 1)] = 0.0
    L[np.diag_indices_from(L)] = np.exp(L[np.diag_indices_from(L)])
    return np.dot(L, L.T)


def check_samples(pot, cov, invcov):
    # NB: the covariance of the generated samples should be the *inverse* of
    # the given covariance because these are momentum samples!
    ndim = len(cov)
    sample_cov0 = np.cov(
        np.random.multivariate_normal(np.zeros(ndim), invcov, size=100000),
        rowvar=0,
    )

    samples = [pot.random() for n in range(10000)]
    sample_cov = np.cov(samples, rowvar=0)
    assert np.all(np.abs(sample_cov - np.eye(ndim)) < 0.1)

    pot.set_var(cov)
    samples = [pot.random() for n in range(100000)]
    sample_cov = np.cov(samples, rowvar=0)
    assert np.all(np.abs(sample_cov - sample_cov0) < 0.1)


def test_draw_samples_diag(ndim=10, seed=9045):
    np.random.seed(seed)
    var = np.random.uniform(0.5, 1.5, ndim)
    pot = WindowedDiagAdapt(ndim)
    check_samples(pot, var, np.diag(1 / var))


def test_draw_samples_full(ndim=10, seed=8976):
    np.random.seed(seed)
    cov = get_cov(ndim)
    cov[np.diag_indices_from(cov)] += 0.1
    invcov = np.linalg.inv(cov)
    pot = WindowedFullAdapt(ndim)
    check_samples(pot, cov, invcov)


def test_sample_p(seed=4566):
    # ref: https://github.com/stan-dev/stan/pull/2672
    np.random.seed(seed)
    m = np.array([[3.0, -2.0], [-2.0, 4.0]])
    m_inv = np.linalg.inv(m)

    var = np.array(
        [
            [2 * m[0, 0], m[1, 0] * m[1, 0] + m[1, 1] * m[0, 0]],
            [m[0, 1] * m[0, 1] + m[1, 1] * m[0, 0], 2 * m[1, 1]],
        ]
    )

    n_samples = 1000
    pot = WindowedFullAdapt(2)
    pot.set_var(m_inv)
    samples = [pot.random() for n in range(n_samples)]
    sample_cov = np.cov(samples, rowvar=0)

    # Covariance matrix within 5 sigma of expected value
    # (comes from a Wishart distribution)
    assert np.all(np.abs(m - sample_cov) < 5 * np.sqrt(var / n_samples))


@pytest.mark.parametrize(
    "pot",
    [
        WindowedDiagAdapt(2, [0, 10]),
        WindowedFullAdapt(2, [0, 10]),
    ],
)
def test_not_invertible(pot):
    for i in range(11):
        pot.update(np.ones(2), None, True)
    with pytest.raises(ValueError):
        pot.raise_ok(None)


@pytest.mark.parametrize(
    "pot",
    [
        WindowedDiagAdapt(2, [10, 15, 36, 64]),
        WindowedFullAdapt(2, [10, 15, 36, 64]),
    ],
)
def test_update_steps(pot):
    np.random.seed(1234)

    # Run through the warm up
    for i in range(pot._update_steps[0]):
        pot.update(np.random.randn(2), None, True)

    # During warmup the variance estimators shouldn't update at all
    assert np.allclose(pot._foreground.n_samples, 0.0)
    assert np.allclose(pot._background.n_samples, 0.0)

    # During the first non-warmup, the the foreground and background should be
    # the same; The -1 here is because the update happens at the end of the
    # step.
    for i in range(pot._update_steps[1] - pot._update_steps[0] - 1):
        pot.update(np.random.randn(2), None, True)
        assert pot._foreground.n_samples > 0
        assert np.allclose(
            pot._foreground.n_samples, pot._background.n_samples
        )

    assert np.allclose(
        pot._foreground.current_variance(), pot._background.current_variance()
    )

    # After hitting the second update_step, the potential variance is updated
    # and the foreground and background are no longer the same. The foreground
    # has been set to the background and the background has been reset.
    pot.update(np.random.randn(2), None, True)
    assert np.allclose(pot._foreground.current_variance(), pot._var)
    assert np.allclose(pot._background.n_samples, 0.0)
    assert np.allclose(
        pot._foreground.n_samples, pot._update_steps[1] - pot._update_steps[0]
    )

    # Finish running all the update_steps
    for i in range(pot._update_steps[-1] - pot._update_steps[1]):
        pot.update(np.random.randn(2), None, True)
    assert np.allclose(pot._background.n_samples, 0.0)
    assert np.allclose(
        pot._foreground.n_samples,
        pot._update_steps[-1] - pot._update_steps[-2],
    )
    assert np.allclose(pot._foreground.current_variance(), pot._var)

    # After finishing all the update steps (during cooldown), the estimates
    # should no longer update
    var = np.copy(pot._var)
    pot.update(np.random.randn(2), None, True)
    assert np.allclose(var, pot._var)
    assert np.allclose(pot._background.n_samples, 0.0)
    assert np.allclose(
        pot._foreground.n_samples,
        pot._update_steps[-1] - pot._update_steps[-2],
    )
    assert np.allclose(pot._foreground.current_variance(), pot._var)


# def test_full_adapt_adaptation_window(seed=8978):
#     np.random.seed(seed)
#     window = 10
#     pot = QuadPotentialDenseAdapt(
#         2, np.zeros(2), np.eye(2), 1, adaptation_window=window
#     )
#     for i in range(window + 1):
#         pot.update(np.random.randn(2), None, True)
#     assert pot._previous_update == window
#     assert pot._adaptation_window == window * 2


# @pytest.mark.filterwarnings("ignore:The number of samples")
# def test_full_adapt_sampling(seed=289586):
#     np.random.seed(seed)

#     L = np.random.randn(5, 5)
#     L[np.diag_indices_from(L)] = np.exp(L[np.diag_indices_from(L)])
#     L[np.triu_indices_from(L, 1)] = 0.0

#     with pm.Model():
#         pm.MvNormal("a", mu=np.zeros(len(L)), chol=L, shape=len(L))
#         sample(draws=10, tune=1000, random_seed=seed, cores=1, chains=1)
