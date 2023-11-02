# -*- coding: utf-8 -*-
# ref: https://github.com/pymc-devs/pymc3/blob/master/pymc3/tests/conftest.py

import pytensor
import pytest


@pytest.fixture(scope="package", autouse=True)
def theano_config():
    flags = dict(compute_test_value="off")
    config = pytensor.config.change_flags(**flags)
    with config:
        yield
