# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

import theano
import theano.tensor as tt
from theano.tests import unittest_tools as utt

from .integrated_limbdark import IntegratedLimbDarkOp


class TestIntegratedLimbDark(utt.InferShapeTester):

    def setUp(self):
        super(TestIntegratedLimbDark, self).setUp()
        self.op_class = IntegratedLimbDarkOp
        self.op = IntegratedLimbDarkOp()

    def get_args(self):
        np.random.seed(1234)

        c = tt.dvector()
        b = tt.dvector()
        r = tt.dvector()
        los = tt.dvector()
        db = tt.dvector()
        d2b = tt.dvector()
        dt = tt.dvector()
        f = theano.function([c, b, r, los, db, d2b, dt],
                            self.op(c, b, r, los, db, d2b, dt)[0])

        c_val = np.array([0.14691226,  0.25709645, -0.01836403])
        b_val = np.linspace(-1.5, 1.5, 100)
        r_val = np.random.uniform(0.01, 0.2, len(b_val))
        los_val = np.ones_like(b_val)
        db_val = np.random.uniform(-10, 10, len(b_val))
        d2b_val = np.random.uniform(-10, 10, len(b_val))
        dt_val = np.random.uniform(0.01, 0.1, len(b_val))

        return (
            f,
            [c, b, r, los, db, d2b, dt],
            [c_val, b_val, r_val, los_val, db_val, d2b_val, dt_val])

    def test_basic(self):
        f, _, in_args = self.get_args()
        out = f(*in_args)
        utt.assert_allclose(0.0, out[0])
        utt.assert_allclose(0.0, out[-1])

    def test_los(self):
        f, _, in_args = self.get_args()
        in_args[3] = -np.ones_like(in_args[-1])
        out = f(*in_args)
        utt.assert_allclose(0.0, out)

    def test_infer_shape(self):
        f, args, arg_vals = self.get_args()
        self._compile_and_check(args,
                                self.op(*args),
                                arg_vals,
                                self.op_class)

    def test_grad(self):
        _, args, in_args = self.get_args()
        func = lambda *args: self.op(*(list(args) + in_args[2:]))[0]  # NOQA
        utt.verify_grad(func, in_args[:2])