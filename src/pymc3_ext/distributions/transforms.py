# -*- coding: utf-8 -*-

__all__ = ["unit_vector", "unit_disk", "angle", "periodic"]

import numpy as np
import pymc3.distributions.transforms as tr
import theano.tensor as tt


class AbsoluteValueTransform(tr.Transform):
    """"""

    name = "absolutevalue"

    def backward(self, y):
        u = 2 * tt.nnet.sigmoid(y) - 1
        return tt.abs_(u)

    def forward(self, x):
        q = 0.5 * (x + 1)
        return tt.log(q) - tt.log(1 - q)

    def forward_val(self, x, point=None):
        q = 0.5 * (x + 1)
        return np.log(q) - np.log(1 - q)

    def jacobian_det(self, y):
        return -2 * tt.nnet.softplus(-y) - y


absolute_value = AbsoluteValueTransform()


class UnitVectorTransform(tr.Transform):
    """A unit vector transformation for PyMC3

    The variable is normalized so that the sum of squares over the last axis
    is unity.

    """

    name = "unitvector"

    def backward(self, y):
        norm = tt.sqrt(tt.sum(tt.square(y), axis=-1, keepdims=True))
        return y / norm

    def forward(self, x):
        return tt.as_tensor_variable(x)

    def forward_val(self, x, point=None):
        return np.copy(x)

    def jacobian_det(self, y):
        return -0.5 * tt.sum(tt.square(y), axis=-1)


unit_vector = UnitVectorTransform()


class UnitDiskTransform(tr.Transform):
    """Transform the 2D real plane into a unit disk

    This will be especially useful for things like sampling in eccentricity
    vectors like ``e sin(w), e cos(w)``.

    """

    name = "unitdisk"

    def backward(self, y):
        return tt.stack([y[0], y[1] * tt.sqrt(1 - y[0] ** 2)])

    def forward(self, x):
        return tt.stack([x[0], x[1] / tt.sqrt(1 - x[0] ** 2)])

    def forward_val(self, x, point=None):
        return np.array([x[0], x[1] / np.sqrt(1 - x[0] ** 2)])

    def jacobian_det(self, y):
        return tt.stack((tt.zeros_like(y[0]), 0.5 * tt.log(1 - y[0] ** 2)))


unit_disk = tr.Chain([UnitDiskTransform(), tr.Interval(-1, 1)])


class AngleTransform(tr.Transform):
    """An angle transformation

    The variable is augmented to sample an isotropic 2D normal and the angle
    is given by the arctan of the ratio of the two coordinates. This will have
    a uniform distribution between -pi and pi.

    Args:
        regularized: The amplitude of the regularization term. If ``None``,
            no regularization is applied. This has no effect on the
            distribution over the transformed parameter, but it can make
            sampling more efficient in some cases.

    """

    name = "angle"

    def __init__(self, *args, **kwargs):
        self.regularized = kwargs.pop("regularized", 10.0)
        super().__init__(*args, **kwargs)

    def backward(self, y):
        return tt.arctan2(y[0], y[1])

    def forward(self, x):
        return tt.concatenate(
            (tt.shape_padleft(tt.sin(x)), tt.shape_padleft(tt.cos(x))), axis=0
        )

    def forward_val(self, x, point=None):
        return np.array([np.sin(x), np.cos(x)])

    def jacobian_det(self, y):
        sm = tt.sum(tt.square(y), axis=0)
        if self.regularized is not None:
            return self.regularized * tt.log(sm) - 0.5 * sm
        return -0.5 * sm


angle = AngleTransform()


class PeriodicTransform(tr.Transform):
    """An periodic transformation

    This extends the :class:`Angle` transform to have a uniform distribution
    between ``lower`` and ``upper``.

    Args:
        lower: The lower bound of the range.
        upper: The upper bound of the range.
        regularized: The amplitude of the regularization term. If ``None``,
            no regularization is applied. This has no effect on the
            distribution over the transformed parameter, but it can make
            sampling more efficient in some cases.

    """

    name = "periodic"

    def __init__(self, lower=0, upper=1, **kwargs):
        self.mid = tt.as_tensor_variable(0.5 * (lower + upper))
        self.delta = tt.as_tensor_variable(0.5 * (upper - lower) / np.pi)
        self.mid_ = 0.5 * (lower + upper)
        self.delta_ = 0.5 * (upper - lower) / np.pi
        self.regularized = kwargs.pop("regularized", 10.0)
        super().__init__(**kwargs)

    def backward(self, y):
        return self.mid + self.delta * tt.arctan2(y[0], y[1])

    def forward(self, x):
        a = (x - self.mid) / self.delta
        return tt.concatenate(
            (tt.shape_padleft(tt.sin(a)), tt.shape_padleft(tt.cos(a))), axis=0
        )

    def forward_val(self, x, point=None):
        a = (x - self.mid_) / self.delta_
        return np.array([np.sin(a), np.cos(a)])

    def jacobian_det(self, y):
        sm = tt.sum(tt.square(y), axis=0)
        if self.regularized is not None:
            return self.regularized * tt.log(sm) - 0.5 * sm
        return -0.5 * sm


periodic = PeriodicTransform
