# -*- coding: utf-8 -*-

import numpy as np
import pymc3 as pm

from pymc3_ext.sampling.sampling import sample


def test_basic():
    np.random.seed(95832)

    with pm.Model():
        pm.Normal("x", shape=5)
        trace = sample(chains=4, draws=1000)
        assert np.all(pm.summary(trace)["r_hat"] < 1.01)

    assert trace["x"].shape == (4000, 5)
