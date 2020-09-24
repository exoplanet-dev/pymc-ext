# -*- coding: utf-8 -*-

import numpy as np
import pymc3 as pm
import pytest
import theano
import theano.tensor as tt
from scipy.stats import invgamma

from pymc3_ext.distributions.helpers import (
    estimate_inverse_gamma_parameters,
    get_log_abs_det_jacobian,
)


def test_get_log_abs_det_jacobian():
    # Sorry this one's a bit convoluted...
    np.random.seed(20200409)

    a = tt.dscalar()
    a.tag.test_value = 0.1
    b = tt.dscalar()
    b.tag.test_value = 0.73452

    c = a + b
    d = a * b

    log_abs_det = get_log_abs_det_jacobian([a, b], [c, d])

    func = theano.function([a, b], tt.stack((c, d, log_abs_det)))
    in_args = [a.tag.test_value, b.tag.test_value]
    grad = []
    for n in range(2):
        grad.append(
            np.append(
                *theano.gradient.numeric_grad(
                    lambda *args: func(*args)[n], in_args
                ).gf
            )
        )

    assert np.allclose(np.linalg.slogdet(grad)[1], func(*in_args)[-1])


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

    samples = pm.InverseGamma.dist(**params).random(size=10000)
    assert np.allclose(
        (samples < lower).sum() / len(samples), target, atol=1e-2
    )
    assert np.allclose(
        (samples > upper).sum() / len(samples), target, atol=1e-2
    )
