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


    def zo_forward(self, ts, x0, p, q):
        self.zo_check(ts, x0, p, q)

        self.xs[0, :] = x0

        for i in range(self.M-1):
            self.update_u(i)
            h = self.ts[i+1] - self.ts[i]

            self.lib.ffcn(self.ts[i:i+1], self.xs[i, :], self.f, self.p, self.u )
            self.xs[i + 1, :]  = self.xs[i,:] +  h*self.f

    def update_u(self, i):
        self.u[:] = self.q[:, i, 0]