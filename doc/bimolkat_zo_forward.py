import numpy
from indegrator.explicit_euler import ExplicitEuler
from indegrator.rk4 import RK4

from indegrator.tapenade import Differentiator
from indegrator.backend_fortran import BackendFortran

d = Differentiator('../examples/fortran/bimolkat/ffcn.f')
backend_fortran = BackendFortran('../examples/fortran/bimolkat/libproblem.so')
rk4 = RK4(backend_fortran)


# =============================================
# zeroth order forward

ts          = numpy.linspace(0,2,50)
x0          = numpy.ones(5)
p           = numpy.ones(5)
q           = numpy.zeros((4, ts.size, 2))
q[0, :, 0]  = 90.
q[1:, :, 0] = 1.

rk4.zo_forward(ts, x0, p, q)
from matplotlib import pyplot
pyplot.figure(figsize=(3, 2))
pyplot.plot(rk4.ts, rk4.xs)
pyplot.xlabel('t')
pyplot.tight_layout()
pyplot.savefig('bimolkat_zo_forward.png')
pyplot.show()

