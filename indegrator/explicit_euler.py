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

from . backend_fortran import BackendFortran
from . backend_pyadolc import BackendPyadolc


class ExplicitEuler(object):

    def __init__(self, backend):

        self.printlevel = 0

        self.NY  = 0           # number of differential variables y
        self.NZ  = 0           # number of algebraic variables z
        self.NX  = 0           # number of variables x = (y,z)
        self.NP  = 0           # number of parameters
        self.NU  = 0           # number of control functions
        self.NQI = 0           # number of q in one control interval

        self.backend = backend

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

        self.P = x0_dot.shape[1]

        assert self.NP == p_dot.shape[0]

        assert self.P == p_dot.shape[1]
        assert self.P == q_dot.shape[3]

        # assign variables
        self.x0_dot = x0_dot
        self.p_dot  = p_dot
        self.q_dot  = q_dot

        # allocate memory
        self.xs_dot = numpy.zeros((self.M, self.NX, self.P))
        self.f_dot  = numpy.zeros((self.NX, self.P))
        self.u_dot  = numpy.zeros((self.NU, self.P))


    def so_check(self, ts,
                 x0, x0_dot2, x0_dot1, x0_ddot,
                 p, p_dot2, p_dot1, p_ddot,
                 q, q_dot2, q_dot1, q_ddot):
        self.zo_check(ts, x0, p, q)

        self.P1 = x0_dot1.shape[1]
        self.P2 = x0_dot2.shape[1]

        assert self.NP == p_dot1.shape[0]
        assert self.NP == p_dot2.shape[0]
        assert self.NP == p_ddot.shape[0]

        assert self.P1 == p_dot1.shape[1]
        assert self.P1 == q_dot1.shape[3]

        assert self.P2 == p_dot2.shape[1]
        assert self.P2 == q_dot2.shape[3]

        assert self.P1 == x0_ddot.shape[1]
        assert self.P1 == p_ddot.shape[1]
        assert self.P1 == q_ddot.shape[3]

        assert self.P2 == x0_ddot.shape[2]
        assert self.P2 == p_ddot.shape[2]
        assert self.P2 == q_ddot.shape[4]


        # assign variables
        self.x0_dot1 = x0_dot1
        self.p_dot1  = p_dot1
        self.q_dot1  = q_dot1

        self.x0_dot2 = x0_dot2
        self.p_dot2  = p_dot2
        self.q_dot2  = q_dot2

        self.x0_ddot = x0_ddot
        self.p_ddot  = p_ddot
        self.q_ddot  = q_ddot

        # allocate memory
        self.xs_dot1 = numpy.zeros((self.M, self.NX, self.P1))
        self.xs_dot2 = numpy.zeros((self.M, self.NX, self.P2))
        self.xs_ddot = numpy.zeros((self.M, self.NX, self.P1, self.P2))

        self.f_dot1  = numpy.zeros((self.NX, self.P1))
        self.f_dot2  = numpy.zeros((self.NX, self.P2))
        self.f_ddot  = numpy.zeros((self.NX, self.P1, self.P2))

        self.u_dot1  = numpy.zeros((self.NU, self.P1))
        self.u_dot2  = numpy.zeros((self.NU, self.P2))
        self.u_ddot  = numpy.zeros((self.NU, self.P1, self.P2))


    def zo_forward(self, ts, x0, p, q):
        self.zo_check(ts, x0, p, q)

        self.xs[0, :] = x0

        for i in range(self.M-1):
            self.update_u(i)
            h = self.ts[i+1] - self.ts[i]

            self.backend.ffcn(self.ts[i:i+1], self.xs[i, :], self.f, self.p, self.u )
            self.xs[i + 1, :]  = self.xs[i,:] +  h*self.f

    def fo_forward_xpu(self, ts, x0, x0_dot, p, p_dot, q, q_dot):
        self.fo_check(ts, x0, x0_dot, p, p_dot, q, q_dot)

        self.xs[0, :]         = x0
        self.xs_dot[0, :, :]  = x0_dot

        for i in range(self.M-1):
            self.update_u_dot(i)
            h = self.ts[i+1] - self.ts[i]

            self.xs_dot[i + 1, :, :]  = self.xs_dot[i,:, :]
            self.xs[i + 1, :]         = self.xs[i, :]

            self.backend.ffcn_dot(self.ts[i:i+1],
                              self.xs[i, :], self.xs_dot[i, :, :],
                              self.f, self.f_dot,
                              self.p, self.p_dot,
                              self.u, self.u_dot)

            self.xs_dot[i + 1, :, :]  += h*self.f_dot
            self.xs[i + 1, :]         += h*self.f

    def so_forward_xpu_xpu(self, ts, x0, x0_dot2, x0_dot1, x0_ddot,
                                     p,   p_dot2,  p_dot1, p_ddot,
                                     q,   q_dot2,  q_dot1, q_ddot):

        self.so_check(ts,
                 x0, x0_dot2, x0_dot1, x0_ddot,
                 p,   p_dot2,  p_dot1, p_ddot,
                 q,   q_dot2,  q_dot1, q_ddot)

        self.xs[0, :]             = x0
        self.xs_dot1[0, :, :]     = x0_dot1
        self.xs_dot2[0, :, :]     = x0_dot2
        self.xs_ddot[0, :, :, :]  = x0_ddot

        for i in range(self.M-1):
            self.update_u_ddot(i)
            h = self.ts[i+1] - self.ts[i]

            self.xs_ddot[i + 1, :, :, :]  = self.xs_ddot[i, :, :, :]
            self.xs_dot1[i + 1, :, :]     = self.xs_dot1[i, :, :]
            self.xs_dot2[i + 1, :, :]     = self.xs_dot2[i, :, :]
            self.xs[i + 1, :]             = self.xs[i, :]

            self.backend.ffcn_ddot(self.ts[i:i+1],
                              self.xs[i, :], self.xs_dot1[i, :, :], self.xs_dot2[i, :, :], self.xs_ddot[i, :, :, :],
                              self.f, self.f_dot1, self.f_dot2, self.f_ddot,
                              self.p, self.p_dot1, self.p_dot2, self.p_ddot,
                              self.u, self.u_dot1, self.u_dot2, self.u_ddot)

            self.xs_ddot[i + 1, :, :, :]  += h*self.f_ddot
            self.xs_dot1[i + 1, :, :]     += h*self.f_dot1
            self.xs_dot2[i + 1, :, :]     += h*self.f_dot2
            self.xs[i + 1, :]             += h*self.f


    def fo_reverse(self, xs_bar):

        numpy.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=200, suppress=None, nanstr=None, infstr=None, formatter=None)
        self.xs_bar = xs_bar.copy()


        self.x0_bar = numpy.zeros(self.x0.shape)
        self.f_bar  = numpy.zeros(self.f.shape)
        self.p_bar  = numpy.zeros(self.p.shape)
        self.q_bar  = numpy.zeros(self.q.shape)
        self.u_bar  = numpy.zeros(self.u.shape)

        for i in range(self.M-1)[::-1]:
            h = self.ts[i+1] - self.ts[i]
            self.update_u(i)

            self.xs_bar[i,:] += self.xs_bar[i + 1, :]

            self.f_bar[:] = h*self.xs_bar[i+1, :]
            self.backend.ffcn_bar(self.ts[i:i+1],
                              self.xs[i, :], self.xs_bar[i,:],
                              self.f, self.f_bar,
                              self.p, self.p_bar,
                              self.u, self.u_bar)

            self.xs_bar[i + 1, :] = 0
            self.update_u_bar(i)

        self.x0_bar[:] += self.xs_bar[0, :]
        self.xs_bar[0, :] = 0.


    def update_u(self, i):
        self.u[:] = self.q[:, i, 0]

    def update_u_dot(self, i):
        self.u[:] = self.q[:, i, 0]
        self.u_dot[:, :] = self.q_dot[:, i, 0, :]

    def update_u_bar(self, i):
        self.q_bar[:, i, 0] = self.u_bar[:]
        self.u_bar[:] = 0.

    def update_u_ddot(self, i):
        self.u[:] = self.q[:, i, 0]
        self.u_dot1[:, :] = self.q_dot1[:, i, 0, :]
        self.u_dot2[:, :] = self.q_dot2[:, i, 0, :]
        self.u_ddot[:, :] = self.q_ddot[:, i, 0, :, :]
