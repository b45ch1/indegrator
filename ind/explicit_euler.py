# Author: Sebastian F. Walter
#         Manuel Kudruss

import os
import sys
import numpy
import json
import datetime
import tempfile
import scipy.linalg
from numpy.testing import assert_almost_equal

from . ffi import libproblem


class ExplicitEuler(object):

    def __init__(self, path_to_libproblem_so):

        self.printlevel = 0

        self.NY  = 0           # number of differential variables y
        self.NZ  = 0           # number of algebraic variables z
        self.NX  = 0           # number of variables x = (y,z)
        self.NP  = 0           # number of parameters
        self.NU  = 0           # number of control functions
        self.NQI = 0           # number of q in one control interval

        self.lib = libproblem(path_to_libproblem_so)

    def zo_check(self, ts, x0, p, q):

        # set dimeions
        self.M   = ts.size              # number of time steps
        self.NQ  = self.NU*self.M*2     # number of control variables

        # assert that the dimensions match

        self.NX = x0.size
        self.NP = p.size

        self.NU = q.shape[0]
        self.M  = q.shape[1]

        # assign variables
        self.ts = ts
        self.x0 = x0
        self.p  = p
        self.q  = q

        # allocate memory
        self.xs  = numpy.zeros((self.M, self.NX))
        self.f   = numpy.zeros(self.NX)
        self.u   = numpy.zeros(self.NU)

    def fo_check(self, ts, x0, x0_dot, p, p_dot, q, q_dot):
        self.zo_check(ts, x0, p, q)

        self.P = x0_dot.shape[0]

        assert self.NP == p_dot.shape[1]

        assert self.P == p_dot.shape[0]
        assert self.P == q_dot.shape[0]

        # assign variables
        self.x0_dot = x0_dot
        self.p_dot  = p_dot
        self.q_dot  = q_dot

        # allocate memory
        self.xs_dot = numpy.zeros((self.M, self.P, self.NX))
        self.f_dot  = numpy.zeros((self.P, self.NX))
        self.u_dot  = numpy.zeros((self.P, self.NU))


    def zo_forward(self, ts, x0, p, q):
        self.zo_check(ts, x0, p, q)

        self.xs[0, :] = x0

        for i in range(self.M-1):
            self.update_u(i)
            h = self.ts[i+1] - self.ts[i]

            self.lib.ffcn(self.ts[i:i+1], self.xs[i, :], self.f, self.p, self.u )
            self.xs[i + 1, :]  = self.xs[i,:] +  h*self.f



    def fo_forward(self, ts, x0, x0_dot, p, p_dot, q, q_dot):
        self.fo_check(ts, x0, x0_dot, p, p_dot, q, q_dot)

        self.xs[0, :]         = x0
        self.xs_dot[0, :, :]  = x0_dot


        for i in range(self.M-1):
            self.update_u_dot(i)
            h = self.ts[i+1] - self.ts[i]

            self.xs_dot[i + 1, :]  = self.xs_dot[i,:]

            self.lib.ffcn_dot(self.ts[i:i+1],
                              self.xs[i, :], self.xs_dot[i, :, :],
                              self.f, self.f_dot,
                              self.p, self.p_dot,
                              self.u, self.u_dot)


            self.xs_dot[i + 1, :, :]  += h*self.f_dot
            self.xs[i + 1, :]  = self.xs[i,:] +  h*self.f


    def update_u(self, i):
        self.u[:] = self.q[:, i, 0]

    def update_u_dot(self, i):
        self.u_dot[:, :] = self.q_dot[:, :, i, 0]