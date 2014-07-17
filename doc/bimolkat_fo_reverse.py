import numpy
from indegrator.explicit_euler import ExplicitEuler
from indegrator.rk4 import RK4

from indegrator.tapenade import Differentiator
from indegrator.ffi import libproblem

d = Differentiator('../examples/bimolkat/ffcn.f')
rk4 = RK4('../examples/bimolkat/libproblem.so')

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

print 'gradient of x(t=2; x0, p, q) w.r.t. p  = \n', rk4.p_bar
print 'gradient of x(t=2; x0, p, q) w.r.t. q  = \n', rk4.q_bar
print 'gradient of x(t=2; x0, p, q) w.r.t. x0 = \n', rk4.x0_bar

