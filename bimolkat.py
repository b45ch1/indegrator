import numpy
from ind.explicit_euler import ExplicitEuler
from ind.tapenade import Differentiator
from ind.ffi import libproblem


# d = Differentiator('./examples/bimolkat/ffcn.f')
mhe = ExplicitEuler('./examples/bimolkat/libproblem.so')

# # =============================================
# # zeroth order forward

# ts          = numpy.linspace(0,2,50)
# x0          = numpy.ones(5)
# p           = numpy.ones(5)
# q           = numpy.zeros((4, ts.size, 2))
# q[0, :, 0]  = 90.
# q[1:, :, 0] = 1.


# mhe.zo_forward(ts, x0, p, q)
# from matplotlib import pyplot
# pyplot.figure()
# pyplot.plot(mhe.ts, mhe.xs)
# pyplot.figure()
# pyplot.plot(mhe.ts, mhe.q[0,:,0])
# pyplot.show()


# =============================================
# first order forward w.r.t. p

ts          = numpy.linspace(0,2,50)
x0          = numpy.ones(5)
p           = numpy.ones(5)
q           = numpy.zeros((4, ts.size, 2))
q[0, :, 0]  = 90.
q[1:, :, 0] = 1.

P           = p.size
x0_dot      = numpy.zeros((x0.size, P))
p_dot       = numpy.zeros((p.size, P))
q_dot       = numpy.zeros(q.shape + (P,))

p_dot[:, :] = numpy.eye(P) 

mhe.fo_forward(ts, x0, x0_dot, p, p_dot, q, q_dot)

print 'sol forward', mhe.xs_dot[-1, 1, :]
# from matplotlib import pyplot
# pyplot.figure()
# pyplot.title('xs')
# pyplot.plot(mhe.ts, mhe.xs)
# pyplot.figure()
# pyplot.title('xs_dot, d/dp1 x_2(t)')
# pyplot.plot(mhe.ts, mhe.xs_dot[:,0,1])
# pyplot.show()


# =============================================
# first order reverse

ts          = numpy.linspace(0,2,50)
x0          = numpy.ones(5)
p           = numpy.ones(5)
q           = numpy.zeros((4, ts.size, 2))
q[0, :, 0]  = 90.
q[1:, :, 0] = 1.

mhe.zo_forward(ts, x0, p, q)




xs_bar = numpy.zeros(mhe.xs.shape)
xs_bar[-1,1] = 1.

mhe.fo_reverse(xs_bar)

print 'sol reverse', mhe.p_bar

# from matplotlib import pyplot
# pyplot.figure()
# pyplot.plot(mhe.ts, mhe.xs)
# pyplot.show()