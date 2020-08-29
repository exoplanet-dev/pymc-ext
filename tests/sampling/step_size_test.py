# -*- coding: utf-8 -*-

import numpy as np

from pymc3_ext.sampling.step_size import WindowedDualAverageAdaptation


def test_update_steps():
    np.random.seed(1234)

    steps = [10, 15, 36, 64]
    step_size = WindowedDualAverageAdaptation(
        steps,
        0.1,
        0.8,
        gamma=0.05,
        k=0.75,
        t0=10,
    )

    # Run through the warm up until the last step
    for i in range(steps[0] - 1):
        step_size.update(np.random.rand(), True)
    assert step_size._count == steps[0]

    # Run the last step of warmup and make sure that the estimator resets
    step_size.update(np.random.rand(), True)
    assert step_size._count == 1
    assert np.allclose(step_size._log_step, np.log(0.1))
    assert step_size._n_samples == steps[0]

    # Run the rest of the warmup and make sure that the estimator is reset
    # correctly
    s0 = steps[0]
    for s in steps[1:]:
        for i in range(s - s0):
            step_size.update(np.random.rand(), True)
        assert step_size._n_samples == s
        assert step_size._count == 1
        assert np.allclose(step_size._log_step, np.log(0.1))
        s0 = s

    # After tuning, the step size should no longer change
    size = step_size.current(False)
    for i in range(10):
        step_size.update(np.random.rand(), False)
    assert np.allclose(step_size.current(False), size)
