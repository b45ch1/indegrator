import os
from numpy.testing import *
import numpy

from indegrator.explicit_euler import ExplicitEuler
from indegrator.rk4 import RK4 

from indegrator.backend_pyadolc import BackendPyadolc


DIR = os.path.dirname(os.path.abspath(__file__))

class Test_RK4(TestCase):


    def test_forward_vs_reverse(self):

        backend_pyadolc = BackendPyadolc(os.path.join(DIR, '../../examples/python/bimolkat/ffcn.py'))
        e = RK4(backend_pyadolc)

        ts          = numpy.linspace(0,2,50)
        x0          = numpy.ones(5)
        p           = numpy.ones(5)
        q           = numpy.zeros((4, ts.size, 2))
        q[0, :, 0]  = 90.
        q[1:, :, 0] = 1.

        P           = 1
        x0_dot      = numpy.zeros((x0.size, P))
        p_dot       = numpy.zeros((p.size, P))
        q_dot       = numpy.zeros(q.shape + (P,))

        p_dot[:, 0] = 1. 

        e.fo_forward_xpu(ts, x0, x0_dot, p, p_dot, q, q_dot)

        xs_bar = numpy.zeros(e.xs.shape)
        xs_bar[-1,1] = 1.

        e.fo_reverse(xs_bar)


        a = numpy.sum(e.x0_bar * e.x0_dot[:,0]) + numpy.sum(e.p_bar * e.p_dot[:,0]) + numpy.sum(e.q_bar * e.q_dot[...,0])
        b = numpy.sum(xs_bar * e.xs_dot[..., 0])

        assert_almost_equal(a, b)


if __name__ == "__main__":
    run_module_suite()



