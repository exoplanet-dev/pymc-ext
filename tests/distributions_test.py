# -*- coding: utf-8 -*-

import logging

import numpy as np
import pymc as pm
import pytest
from scipy.stats import kstest

from pymc_ext.distributions import angle, unit_disk


class _Base:
    random_seed = 20160911

    def _sample(self, **kwargs):
        logger = logging.getLogger("pymc3")
        logger.propagate = False
        logger.setLevel(logging.ERROR)
        kwargs["draws"] = kwargs.get("draws", 1000)
        kwargs["progressbar"] = kwargs.get("progressbar", False)
        kwargs["return_inferencedata"] = kwargs.get(
            "return_inferencedata", False
        )
        return pm.sample(**kwargs)

    def _model(self, **kwargs):
        np.random.seed(self.random_seed)
        return pm.Model(**kwargs)


class TestBase(_Base):
    def test_unit_disk(self):
        with self._model():
            unit_disk("x", "y", shape=(3,), initval=0.01 * np.ones((2, 3)))
            trace = self._sample()

        theta = np.arctan2(trace["x"], trace["y"])
        radius = trace["x"] ** 2 + trace["y"] ** 2

        # Make sure that the unit constraint is satisfied
        assert np.all(radius <= 1.0)

        # The angle should be uniformly distributed
        cdf = lambda x: np.clip((x + np.pi) / (2 * np.pi), 0, 1)  # NOQA
        for i in range(theta.shape[1]):
            s, _ = kstest(theta[:, i], cdf)
            assert s < 0.05

        # As should the radius component
        cdf = lambda x: np.clip(x, 0, 1)  # NOQA
        for i in range(radius.shape[1]):
            s, p = kstest(radius[:, i], cdf)
            assert s < 0.05

    @pytest.mark.parametrize("regularization", [None, 10.0])
    def test_angle(self, regularization):
        with self._model():
            angle("theta", shape=(5, 2), regularization=regularization)
            trace = self._sample(draws=2000)

        # The angle should be uniformly distributed
        theta = trace["theta"]
        theta = np.reshape(theta, (len(theta), -1))
        cdf = lambda x: np.clip((x + np.pi) / (2 * np.pi), 0, 1)  # NOQA
        for i in range(theta.shape[1]):
            s, _ = kstest(theta[:, i], cdf)
            assert s < 0.05
