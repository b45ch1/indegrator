import numpy
from ind.explicit_euler import ExplicitEuler
from ind.tapenade import Differentiator
from ind.ffi import libproblem


# d = Differentiator('./examples/bimolkat/ffcn.f')
# l = libproblem('./examples/bimolkat/libproblem.so')

# t           = numpy.zeros(1)
# x           = numpy.ones(5)
# f           = numpy.ones(5)
# p           = numpy.ones(5)
# u           = numpy.array([90., 1., 1., 1.])


# l.ffcn(t, x, f, p, u)

# print f


mhe = ExplicitEuler('./examples/bimolkat/libproblem.so')

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



# first order forward w.r.t. p

ts          = numpy.linspace(0,2,50)
x0          = numpy.ones(5)
p           = numpy.ones(5)
q           = numpy.zeros((4, ts.size, 2))
q[0, :, 0]  = 90.
q[1:, :, 0] = 1.

P           = p.size
x0_dot      = numpy.zeros((P, x0.size))
p_dot       = numpy.zeros((P, p.size))
q_dot       = numpy.zeros((P,) + q.shape)

p_dot[:, :] = numpy.eye(p.size)


mhe.fo_forward(ts, x0, x0_dot, p, p_dot, q, q_dot)
from matplotlib import pyplot
pyplot.figure()
pyplot.plot(mhe.ts, mhe.xs)
pyplot.figure()
pyplot.plot(mhe.ts, mhe.xs_dot[:,0,:])
pyplot.figure()
pyplot.plot(mhe.ts, mhe.q[0,:,0])
pyplot.show()