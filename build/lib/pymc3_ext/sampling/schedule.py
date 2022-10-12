# -*- coding: utf-8 -*-

__all__ = ["build_schedule"]

import warnings

import numpy as np


def build_schedule(
    tune, *, warmup_window=50, adapt_window=50, cooldown_window=50
):
    tune = int(tune)
    warmup_window = int(warmup_window)
    adapt_window = int(adapt_window)
    cooldown_window = int(cooldown_window)

    if tune < 0:
        raise ValueError("'tune' must be >=0")
    if warmup_window < 0:
        raise ValueError("'warmup_window' must be >=0")
    if adapt_window < 1:
        raise ValueError("'adapt_window' must be >=1")
    if cooldown_window < 0:
        raise ValueError("'cooldown_window' must be >=0")

    # Special cases when tune is too small even for the hack below
    if tune == 0:
        warnings.warn("with zero tuning samples, the schedule is empty")
        return np.array([], dtype=int)

    if warmup_window + adapt_window + cooldown_window > tune:
        warnings.warn(
            "there are not enough tuning steps to accomodate the tuning "
            "schedule; assigning automatically as 20%/70%/10%"
        )
        warmup_window = np.ceil(0.2 * tune).astype(int)
        cooldown_window = np.ceil(0.1 * tune).astype(int)
        adapt_window = tune - warmup_window - cooldown_window

        # If this didn't cut it, 'tune' is too small (this should only happen
        # when tune == 1) just return one step of tuning
        if adapt_window <= 0:
            warmup_window = 0
            cooldown_window = 0
            adapt_window = tune

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
