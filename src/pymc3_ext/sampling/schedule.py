# -*- coding: utf-8 -*-

__all__ = ["build_schedule"]

import numpy as np

from ..utils import logger


def build_schedule(
    tune, warmup_window=50, adapt_window=50, cooldown_window=50
):
    if warmup_window + adapt_window + cooldown_window > tune:
        logger.warn(
            "there are not enough tuning steps to accomodate the tuning "
            "schedule; assigning automatically as 20%/70%/10%"
        )
        warmup_window = np.ceil(0.2 * tune).astype(int)
        cooldown_window = np.ceil(0.1 * tune).astype(int)
        adapt_window = tune - warmup_window - cooldown_window

    t = warmup_window
    delta = adapt_window
    update_steps = []
    while t < tune - cooldown_window:
        t += delta
        delta = 2 * delta
        if t + delta > tune - cooldown_window:
            update_steps.append(tune - cooldown_window)
            break
        update_steps.append(t)

    update_steps = np.array(update_steps, dtype=int)
    if np.any(update_steps) <= 0:
        raise ValueError("invalid tuning schedule")

    return np.append(warmup_window, update_steps)
