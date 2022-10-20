# -*- coding: utf-8 -*-
# ref: https://github.com/pymc-devs/pymc3/blob/master/pymc3/tests/conftest.py

import aesara
import pytest


@pytest.fixture(scope="package", autouse=True)
def theano_config():
    flags = dict(compute_test_value="off")
    config = aesara.configparser.change_flags(**flags)
    with config:
        yield
