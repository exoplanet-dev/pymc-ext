__all__ = ["angle", "unit_disk"]

import warnings

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def angle(name, *, regularization=10.0, **kwargs):
    """An angle constrained to be in the range -pi to pi

    The actual sampling is performed in the two dimensional vector space
    proportional to ``(sin(theta), cos(theta))`` so that the sampler doesn't see
    a discontinuity at pi.

    The regularization parameter can be used to improve sampling performance
    when the value of the angle is well constrained. It removes prior mass near
    the origin in the sampling space, which can lead to bad geometry when the
    angle is poorly constrained, but better performance when it is. The default
    value of ``10.0`` is a good starting point.
    """
    shape = kwargs.get("shape", ())
    initval = kwargs.pop("initval", pt.broadcast_to(0.0, shape))
    x1 = pm.Normal(f"__{name}_angle1", initval=np.sin(initval), **kwargs)
    x2 = pm.Normal(f"__{name}_angle2", initval=np.cos(initval), **kwargs)
    if regularization is not None:
        pm.Potential(
            f"__{name}_regularization",
            regularization * pt.log(x1**2 + x2**2),
        )
    return pm.Deterministic(name, pt.arctan2(x1, x2))


def unit_disk(name_x, name_y, **kwargs):
    """Two dimensional parameters constrained to live within the unit disk

    This returns two distributions whose sum of squares will be in the range
    ``[0, 1)``. For example, in this code block:

    .. code-block:: python

        x, y = unit_disk("x", "y") radius_sq = x**2 + y**2

    the tensor ``radius_sq`` will always have a value in the range ``[0, 1)``.

    Args:
        name_x: The name of the first distribution.
        name_y: The name of the second distribution.
    """
    initval = kwargs.pop("initval", [0.0, 0.0])
    kwargs["lower"] = -1.0
    kwargs["upper"] = 1.0
    x1 = pm.Uniform(name_x, initval=initval[0], **kwargs)
    x2 = pm.Uniform(
        f"__{name_y}_unit_disk",
        initval=initval[1] * np.sqrt(1 - initval[0] ** 2),
        **kwargs,
    )
    norm = pt.sqrt(1 - x1**2)
    pm.Potential(f"__{name_y}_jacobian", pt.log(norm))
    return x1, pm.Deterministic(name_y, x2 * norm)
