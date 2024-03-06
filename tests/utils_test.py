import numpy as np
import pymc as pm
import pytest
from scipy.stats import invgamma

from pymc_ext.utils import estimate_inverse_gamma_parameters, eval_in_model


def test_eval_in_model(seed=123409):
    np.random.seed(seed)
    x_val = np.random.randn(5, 3)
    x_val2 = np.random.randn(5, 3)
    with pm.Model():
        x = pm.Normal("x", shape=x_val.shape, initval=x_val)
        assert np.allclose(eval_in_model(x), x_val)
        assert np.allclose(eval_in_model(x, {"x": x_val2}), x_val2)


def test_eval_in_model_uniform(seed=123409):
    # test_eval_in_model has unconstrained (-inf, inf) variables only
    # Uniform has implicit transform in PyMC so check that this works too with
    # eval_in_model
    rng = np.random.default_rng(seed)
    x_val = rng.uniform(size=(5, 3))
    with pm.Model():
        x = pm.Uniform("x", shape=x_val.shape, initval=x_val)

        assert np.allclose(eval_in_model(x), x_val)


def test_eval_in_model_list(seed=123409):
    # The utils.Evaluator class handles list of variables differently
    # from single variables, so we test this here.
    rng = np.random.default_rng(seed)
    x_val = rng.uniform(size=(5, 3))
    y_val = rng.standard_normal()

    with pm.Model():
        x = pm.Uniform("x", shape=x_val.shape, initval=x_val)
        y = pm.Normal("y", initval=y_val)
        x_eval, y_eval = eval_in_model([x, y])
        assert np.allclose(x_eval, x_val)
        assert np.allclose(y_eval, y_val)


@pytest.mark.parametrize(
    "lower, upper, target",
    [(1.0, 2.0, 0.01), (0.01, 0.1, 0.1), (10.0, 25.0, 0.01)],
)
def test_estimate_inverse_gamma_parameters(lower, upper, target):
    np.random.seed(20200409)

    params = estimate_inverse_gamma_parameters(lower, upper, target=target)
    dist = invgamma(params["alpha"], scale=params["beta"])
    assert np.allclose(dist.cdf(lower), target)
    assert np.allclose(1 - dist.cdf(upper), target)

    samples = pm.draw(pm.InverseGamma.dist(**params), draws=10000)
    assert np.allclose(
        (samples < lower).sum() / len(samples), target, atol=1e-2
    )
    assert np.allclose(
        (samples > upper).sum() / len(samples), target, atol=1e-2
    )
