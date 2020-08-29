# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pymc3_ext.sampling.schedule import build_schedule


@pytest.mark.parametrize("warmup", [0, 5, 10, 50])
@pytest.mark.parametrize("adapt", [1, 6, 11, 51])
@pytest.mark.parametrize("cooldown", [0, 4, 9, 49])
def test_basic(warmup, adapt, cooldown):
    tune = 1000
    update_steps = build_schedule(
        tune,
        warmup_window=warmup,
        adapt_window=adapt,
        cooldown_window=cooldown,
    )

    # Sorted - not necessary, but nice!
    assert np.all(np.diff(update_steps) > 0)

    # These ones are trivial
    assert update_steps[0] == warmup
    assert update_steps[1] == warmup + adapt
    assert update_steps[-1] == tune - cooldown

    # Check that the window is doubling appropriately
    assert np.allclose(
        np.diff(update_steps[1:-1]) / adapt,
        2 ** np.arange(1, len(update_steps) - 2),
    )

    # The last window should always be at least as long as the doubling would
    # expect, never shorter
    assert (update_steps[-1] - update_steps[-2]) / adapt >= 2 ** (
        len(update_steps) - 2
    )


@pytest.mark.parametrize(
    "tune, warmup, adapt, cooldown",
    [
        (100, 49, 50, 51),
        (0, 49, 50, 51),
        (1, 49, 50, 51),
        (2, 49, 50, 51),
    ],
)
def test_too_short(tune, warmup, adapt, cooldown):
    with pytest.warns(UserWarning):
        update_steps = build_schedule(
            tune,
            warmup_window=warmup,
            adapt_window=adapt,
            cooldown_window=cooldown,
        )
    if tune == 0:
        assert len(update_steps) == 0
    else:
        assert np.all(update_steps[1:] > 0)
        assert np.all(update_steps <= tune)
