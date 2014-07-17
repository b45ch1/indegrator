import numpy
from indegrator.explicit_euler import ExplicitEuler
from indegrator.rk4 import RK4

from indegrator.tapenade import Differentiator
from indegrator.ffi import libproblem

import time

# d = Differentiator('./examples/bimolkat/ffcn.f')
# exit()
rk4 = ExplicitEuler('./examples/bimolkat/libproblem.so')
# rk4 = RK4('./examples/bimolkat/libproblem.so')

# # =============================================
# # zeroth order forward

# ts          = numpy.linspace(0,2,50)
# x0          = numpy.ones(5)
# p           = numpy.ones(5)
# q           = numpy.zeros((4, ts.size, 2))
# q[0, :, 0]  = 90.
# q[1:, :, 0] = 1.


# rk4.zo_forward(ts, x0, p, q)
# from matplotlib import pyplot
# pyplot.figure()
# pyplot.plot(rk4.ts, rk4.xs)
# pyplot.figure()
# pyplot.plot(rk4.ts, rk4.q[0,:,0])
# pyplot.show()


# # =============================================
# # first order forward w.r.t. p

# ts          = numpy.linspace(0,2,5000)
# x0          = numpy.ones(5)
# p           = numpy.ones(5)
# q           = numpy.zeros((4, ts.size, 2))
# q[0, :, 0]  = 90.
# q[1:, :, 0] = 1.

# P           = p.size
# x0_dot      = numpy.zeros((x0.size, P))
# p_dot       = numpy.zeros((p.size, P))
# q_dot       = numpy.zeros(q.shape + (P,))

# p_dot[:, :] = numpy.eye(P) 

# st = time.time()
# rk4.fo_forward_xpu(ts, x0, x0_dot, p, p_dot, q, q_dot)
# print 'dt=', time.time() - st

# from matplotlib import pyplot
# pyplot.figure()
# pyplot.title('xs')
# pyplot.plot(rk4.ts, rk4.xs)
# pyplot.figure()
# pyplot.title('xs_dot, d/dp1 x_2(t)')
# pyplot.plot(rk4.ts, rk4.xs_dot[:,1,0])
# pyplot.show()


# # =============================================
# # first order forward w.r.t. 1

# ts          = numpy.linspace(0,2,30)
# x0          = numpy.ones(5)
# p           = numpy.ones(5)
# q           = numpy.zeros((4, ts.size, 1))
# q[0, :]     = 90.
# q[1:, :]    = 1.

# P           = q.size
# x0_dot      = numpy.zeros((x0.size, P))
# p_dot       = numpy.zeros((p.size, P))
# q_dot       = numpy.zeros(q.shape + (P,))

# q_dot.reshape((P, P))[:, :] = numpy.eye(P) 

# st = time.time()
# rk4.fo_forward_xpu(ts, x0, x0_dot, p, p_dot, q, q_dot)
# print 'dt=', time.time() - st

# from matplotlib import pyplot
# pyplot.figure()
# pyplot.title('xs')
# pyplot.plot(rk4.ts, rk4.xs)
# pyplot.figure()
# pyplot.title('d/dq x(t)')
# pyplot.plot(rk4.ts, rk4.xs_dot[:,1,:30])
# print rk4.xs_dot[:,1,:30]
# pyplot.show()



# =============================================
# first order reverse

ts          = numpy.linspace(0,2,500)
x0          = numpy.ones(5)
p           = numpy.ones(5)
q           = numpy.zeros((4, ts.size, 2))
q[0, :, 0]  = 90.
q[1:, :, 0] = 1.

rk4.zo_forward(ts, x0, p, q)




xs_bar = numpy.zeros(rk4.xs.shape)
xs_bar[-1,1] = 1.

rk4.fo_reverse(xs_bar)

print 'sol reverse', rk4.p_bar
print 'sol reverse', rk4.q_bar


# print rk4.xs_bar

# from matplotlib import pyplot
# pyplot.figure()
# pyplot.plot(rk4.ts, rk4.xs)
# pyplot.show()





# # =============================================
# # second-order forward w.r.t. p and q

# ts           = numpy.linspace(0,2,1000)
# x0           = numpy.ones(5)
# p            = numpy.ones(5)
# q            = numpy.zeros((4, ts.size, 2))
# q[0, :, 0]   = 90.
# q[1:, :, 0]  = 1.

# P1           = p.size
# x0_dot1      = numpy.zeros((x0.size, P1))
# p_dot1       = numpy.zeros((p.size, P1))
# q_dot1       = numpy.zeros(q.shape + (P1,))
# p_dot1[:, :] = numpy.eye(P1) 

# P2           = 1
# x0_dot2      = numpy.zeros((x0.size, P2))
# p_dot2       = numpy.zeros((p.size, P2))
# q_dot2       = numpy.zeros(q.shape + (P2,))
# q_dot2[0, 0, 0, 0] = 1

# x0_ddot      = numpy.zeros((x0.size, P1, P2))
# p_ddot       = numpy.zeros((p.size, P1, P2))
# q_ddot       = numpy.zeros(q.shape + (P1, P2))

# st = time.time()
# rk4.so_forward_xpu_xpu(ts,
#                        x0, x0_dot2, x0_dot1, x0_ddot,
#                        p,   p_dot2,  p_dot1,  p_ddot,
#                        q,   q_dot2,  q_dot1,  q_ddot)

# print 'dt=', time.time() - st

# from matplotlib import pyplot
# pyplot.figure()
# pyplot.title('xs')
# pyplot.plot(rk4.ts, rk4.xs)
# pyplot.figure()
# pyplot.title('xs_dot, d/dp1 x_2(t)')
# pyplot.plot(rk4.ts, rk4.xs_dot1[:,1,0])

# pyplot.figure()
# pyplot.title('xs_dot, d/dq1 x_2(t)')
# pyplot.plot(rk4.ts, rk4.xs_dot2[:,1,0])

# pyplot.figure()
# pyplot.title('xs_dot, d/dp1 d/dq1 x_2(t)')
# pyplot.plot(rk4.ts, rk4.xs_ddot[:,1,0,0])


# pyplot.show()