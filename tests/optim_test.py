import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from pymc_ext.optim import optimize


def test_optimize(seed=1234):
    random = np.random.default_rng(seed)
    x_val = random.standard_normal(size=(5, 3))
    with pm.Model():
        x = pm.Normal("x", shape=x_val.shape, initval=x_val)
        y = pm.Normal("y", shape=x_val.shape, initval=x_val)
        soln1 = optimize()
        soln2 = optimize(vars=[x + y])

    assert np.allclose(soln1["x"], 0.0)
    assert np.allclose(soln1["y"], 0.0)
    assert np.allclose(soln2["x"], 0.0)
    assert np.allclose(soln2["y"], 0.0)
