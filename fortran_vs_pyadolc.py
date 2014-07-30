import numpy
from indegrator.explicit_euler import ExplicitEuler
from indegrator.rk4 import RK4

from indegrator.tapenade import Differentiator
from indegrator.backend_fortran import BackendFortran
from indegrator.backend_pyadolc import BackendPyadolc

import time

from numpy.testing import *

# d = Differentiator('./examples/fortran/bimolkat/ffcn.f')
backend_fortran = BackendFortran('./examples/fortran/bimolkat/libproblem.so')
backend_pyadolc = BackendPyadolc('./examples/python/bimolkat/ffcn.py')

rk4f = RK4(backend_fortran)
rk4p = RK4(backend_pyadolc)


STEPS = 2000


print 'zo forward'
ts          = numpy.linspace(0,2,STEPS)
x0          = numpy.ones(5)
p           = numpy.ones(5)
q           = numpy.zeros((4, ts.size, 2))
q[0, :, 0]  = 90.
q[1:, :, 0] = 1.


rk4f.zo_forward(ts, x0, p, q)
tic = time.time()
rk4f.zo_forward(ts, x0, p, q)
toc = time.time()
print 'fortran time = ', toc - tic


tic = time.time()
rk4p.zo_forward(ts, x0, p, q)
toc = time.time()
print 'pyadolc time = ', toc - tic




print 'fo forward'

ts          = numpy.linspace(0,2,STEPS)
x0          = numpy.ones(5)
p           = numpy.ones(5)
q           = numpy.zeros((4, ts.size, 2))
q[0, :, 0]  = 90.
q[1:, :, 0] = 1.

P           = p.size
x0_dot      = numpy.zeros((x0.size, P))
p_dot       = numpy.zeros((p.size, P))
q_dot       = numpy.zeros(q.shape + (P,))
p_dot[:, 0] = 1.

tic = time.time()
rk4f.fo_forward_xpu(ts,
                      x0, x0_dot,
                      p, p_dot, 
                      q, q_dot) 


toc = time.time()
print 'fortran time = ', toc - tic


tic = time.time()
rk4p.fo_forward_xpu(ts,
                      x0, x0_dot,
                      p, p_dot, 
                      q, q_dot) 

toc = time.time()
print 'pyadolc time = ', toc - tic


print 'so forward'

ts          = numpy.linspace(0,2,20)
x0          = numpy.ones(5)
p           = numpy.ones(5)
q           = numpy.zeros((4, ts.size, 2))
q[0, :, 0]  = 90.
q[1:, :, 0] = 1.

P1           = p.size
x0_dot1      = numpy.zeros((x0.size, P1))
p_dot1       = numpy.zeros((p.size, P1))
q_dot1       = numpy.zeros(q.shape + (P1,))
p_dot1[:, 0] = 1.

P2           = q.size
x0_dot2      = numpy.zeros((x0.size, P2))
p_dot2       = numpy.zeros((p.size, P2))
q_dot2       = numpy.zeros(q.shape + (P2,))
q_dot2[:, 0] = 1.


x0_ddot      = numpy.zeros((x0.size, P1, P2))
p_ddot       = numpy.zeros((p.size, P1, P2))
q_ddot       = numpy.zeros(q.shape + (P1, P2))


tic = time.time()
rk4f.so_forward_xpu_xpu(ts,
                      x0, x0_dot2, x0_dot1, x0_ddot,
                      p, p_dot2, p_dot1, p_ddot,
                      q, q_dot2, q_dot1, q_ddot)


toc = time.time()
print 'fortran time = ', toc - tic


tic = time.time()
rk4p.so_forward_xpu_xpu(ts,
                      x0, x0_dot2, x0_dot1, x0_ddot,
                      p, p_dot2, p_dot1, p_ddot,
                      q, q_dot2, q_dot1, q_ddot)

toc = time.time()
print 'pyadolc time = ', toc - tic




# xs_bar = numpy.zeros(ee.xs.shape)
# xs_bar[-1,1] = 1.

# ee.fo_reverse(xs_bar)


# rk4 = RK4(backend_fortran)
# ts          = numpy.linspace(0,2,STEPS)
# x0          = numpy.ones(5)
# p           = numpy.ones(5)
# q           = numpy.zeros((4, ts.size, 2))
# q[0, :, 0]  = 90.
# q[1:, :, 0] = 1.

# rk4.zo_forward(ts, x0, p, q)

# xs_bar = numpy.zeros(rk4.xs.shape)
# xs_bar[-1,1] = 1.

# rk4.fo_reverse(xs_bar)

# print rk4.x0_bar -  ee.x0_bar


# ee = ExplicitEuler(backend_fortran)
# ts          = numpy.linspace(0,2,STEPS)
# x0          = numpy.ones(5)
# p           = numpy.ones(5)
# q           = numpy.zeros((4, ts.size, 2))
# q[0, :, 0]  = 90.
# q[1:, :, 0] = 1.

# P           = 1
# x0_dot      = numpy.zeros((x0.size, P))
# p_dot       = numpy.zeros((p.size, P))
# q_dot       = numpy.zeros(q.shape + (P,))

# p_dot[:, 0] = 1. 

# ee.fo_forward_xpu(ts, x0, x0_dot, p, p_dot, q, q_dot)




# rk4 = RK4(backend_fortran)
# ts          = numpy.linspace(0,2,STEPS)
# x0          = numpy.ones(5)
# p           = numpy.ones(5)
# q           = numpy.zeros((4, ts.size, 2))
# q[0, :, 0]  = 90.
# q[1:, :, 0] = 1.

# P           = 1
# x0_dot      = numpy.zeros((x0.size, P))
# p_dot       = numpy.zeros((p.size, P))
# q_dot       = numpy.zeros(q.shape + (P,))

# p_dot[:, 0] = 1. 

# rk4.fo_forward_xpu(ts, x0, x0_dot, p, p_dot, q, q_dot)

# print rk4.xs_dot - ee.xs_dot





# e = RK4(backend_fortran)
# # e = ExplicitEuler(backend_fortran)

# ts          = numpy.linspace(0,0.1,100)
# x0          = numpy.ones(5)
# p           = numpy.ones(5)
# q           = numpy.zeros((4, ts.size, 2))
# q[0, :, 0]  = 90.
# q[1:, :, 0] = 1.

# P           = 1
# x0_dot      = numpy.zeros((x0.size, P))
# p_dot       = numpy.zeros((p.size, P))
# q_dot       = numpy.zeros(q.shape + (P,))

# p_dot[:, 0] = 1. 

# e.fo_forward_xpu(ts, x0, x0_dot, p, p_dot, q, q_dot)

# xs_bar = numpy.zeros(e.xs.shape)
# xs_bar[-1,1] = 1.

# e.fo_reverse(xs_bar)

# a = numpy.sum(e.x0_bar * e.x0_dot[:,0]) + numpy.sum(e.p_bar * e.p_dot[:,0]) + numpy.sum(e.q_bar * e.q_dot[...,0])
# b = numpy.sum(xs_bar * e.xs_dot[..., 0])


# # print numpy.sum(e.x0_bar * e.x0_dot[:,0])
# # print numpy.sum(e.p_bar * e.p_dot[:,0])
# # print numpy.sum(e.q_bar * e.q_dot[...,0])
# # print numpy.sum(xs_bar * e.xs_dot[..., 0])

# assert_almost_equal(a, b)














# STEPS = 200

# e = RK4(backend_fortran)

# ts          = numpy.linspace(0,2,STEPS)
# x0          = numpy.ones(5)
# p           = numpy.ones(5)
# q           = numpy.zeros((4, ts.size, 2))
# q[0, :, 0]  = 90.
# q[1:, :, 0] = 1.
# e.zo_forward(ts, x0, p, q)

# a = e.xs.copy()


# # e = ExplicitEuler(backend_fortran)

# ts          = numpy.linspace(0,2,STEPS)
# x0          = numpy.ones(5)
# p           = numpy.ones(5)
# q           = numpy.zeros((4, ts.size, 2))
# q[0, :, 0]  = 90.
# q[1:, :, 0] = 1.

# P           = 1
# x0_dot      = numpy.zeros((x0.size, P))
# p_dot       = numpy.zeros((p.size, P))
# q_dot       = numpy.zeros(q.shape + (P,))

# p_dot[:, 0] = 1. 

# e.fo_forward_xpu(ts, x0, x0_dot, p, p_dot, q, q_dot)

# b  = e.xs.copy()

# print b - a