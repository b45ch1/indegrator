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

    def test_fortran_vs_pyadolc_so_forward(self):

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

        P1           = 1
        x0_dot1      = numpy.zeros((x0.size, P1))
        p_dot1       = numpy.zeros((p.size, P1))
        q_dot1       = numpy.zeros(q.shape + (P1,))
        p_dot1[:, 0] = 1.

        P2           = 1
        x0_dot2      = numpy.zeros((x0.size, P2))
        p_dot2       = numpy.zeros((p.size, P2))
        q_dot2       = numpy.zeros(q.shape + (P2,))
        q_dot2[:, 0] = 1.


        x0_ddot      = numpy.zeros((x0.size, P1, P2))
        p_ddot       = numpy.zeros((p.size, P1, P2))
        q_ddot       = numpy.zeros(q.shape + (P1, P2))

        ef.so_forward_xpu_xpu(ts,
                              x0, x0_dot1, x0_dot2, x0_ddot,
                              p, p_dot1, p_dot2, p_ddot,
                              q, q_dot1, q_dot2, q_ddot)

        xs_fortran = ef.xs.copy()
        xs_dot1_fortran = ef.xs_dot1.copy()
        xs_dot2_fortran = ef.xs_dot2.copy()
        xs_ddot_fortran = ef.xs_ddot.copy()


        ep.so_forward_xpu_xpu(ts,
                              x0, x0_dot1, x0_dot2, x0_ddot,
                              p, p_dot1, p_dot2, p_ddot,
                              q, q_dot1, q_dot2, q_ddot)


        xs_pyadolc = ep.xs.copy()
        xs_dot1_pyadolc = ep.xs_dot1.copy()
        xs_dot2_pyadolc = ep.xs_dot2.copy()
        xs_ddot_pyadolc = ep.xs_ddot.copy()

        assert_almost_equal(xs_fortran, xs_pyadolc)
        assert_almost_equal(xs_dot1_fortran, xs_dot1_pyadolc)
        assert_almost_equal(xs_dot2_fortran, xs_dot2_pyadolc)
        assert_almost_equal(xs_ddot_fortran, xs_ddot_pyadolc)




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



