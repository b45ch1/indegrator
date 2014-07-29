import os
from numpy.testing import *
import numpy

from indegrator.explicit_euler import ExplicitEuler 
from indegrator.tapenade import Differentiator 
from indegrator.backend_fortran import BackendFortran
from indegrator.backend_pyadolc import BackendPyadolc


from numpy.testing import *


DIR = os.path.dirname(os.path.abspath(__file__))

class Test_ExplicitEuler(TestCase):


    def test_forward_vs_reverse(self):

        d = Differentiator(os.path.join(DIR, '../../examples/fortran/bimolkat/ffcn.f'))
        backend_fortran = BackendFortran(os.path.join(DIR, '../../examples/fortran/bimolkat/libproblem.so'))
        e = ExplicitEuler(backend_fortran)

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

    def test_fortran_vs_pyadolc_fo_forward(self):

        # d = Differentiator(os.path.join(DIR, '../../examples/fortran/bimolkat/ffcn.f'))
        backend_fortran = BackendFortran(os.path.join(DIR, '../../examples/fortran/bimolkat/libproblem.so'))
        backend_pyadolc = BackendPyadolc(os.path.join(DIR, '../../examples/python/bimolkat/ffcn.py'))
        ef = ExplicitEuler(backend_fortran)
        ep = ExplicitEuler(backend_pyadolc)

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

        ef.fo_forward_xpu(ts, x0, x0_dot, p, p_dot, q, q_dot)
        xs_fortran = ef.xs.copy()
        xs_dot_fortran = ef.xs_dot.copy()

        ep.fo_forward_xpu(ts, x0, x0_dot, p, p_dot, q, q_dot)
        xs_pyadolc = ep.xs.copy()
        xs_dot_pyadolc = ep.xs_dot.copy()
      
        assert_almost_equal(xs_fortran, xs_pyadolc) 
        assert_almost_equal(xs_dot_fortran, xs_dot_pyadolc) 


    def test_pyadolc_fo_reverse(self):

        backend_pyadolc = BackendPyadolc(os.path.join(DIR, '../../examples/python/bimolkat/ffcn.py'))
        e = ExplicitEuler(backend_pyadolc)

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



