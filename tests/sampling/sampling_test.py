# -*- coding: utf-8 -*-

import numpy as np
import pymc3 as pm

from pymc3_ext.sampling.sampling import sample
from pymc3_ext.sampling.groups import ParameterGroup


def test_basic():
    np.random.seed(95832)

    with pm.Model():
        pm.Normal("x", shape=5)
        trace = sample(chains=4, draws=1000, progressbar=False)
        assert np.all(pm.summary(trace)["r_hat"] < 1.01)

    assert trace["x"].shape == (4000, 5)


def test_correlated_groups():
    np.random.seed(42)
    ndim = 5
    L = np.random.randn(ndim, ndim)
    L[np.diag_indices_from(L)] = 0.1 * np.exp(L[np.diag_indices_from(L)])
    L[np.triu_indices_from(L, 1)] = 0.0

    with pm.Model() as model:
        pm.MvNormal("x", mu=np.zeros(ndim), chol=L, shape=ndim)
        pm.MvNormal("y", mu=np.zeros(ndim), chol=L, shape=ndim)
        pm.Normal("z", shape=ndim)  # Uncorrelated
        trace = sample(
            tune=1000,
            draws=2000,
            chains=4,
            parameter_groups=[
                [model.x],
                [model.y],
                ParameterGroup([model.z], "diag"),
            ],
            progressbar=False,
        )
        assert np.all(pm.summary(trace)["r_hat"] < 1.01)
