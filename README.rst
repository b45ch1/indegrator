INDegrator, an internal numerical differentiation (IND) library of differentiated ODE solvers
---------------------------------------------------------------------------------------------


Description:

    INDegrator is a library of IND integration schemes. 
    They allow you to evaluate the solution ``y(t; y0, p, q)`` of initial value
    problem (IVP) of the form::


        y_t  = f(y, p, q)
        y(0) = y0

    where y_t denotes the derivative of ``y`` w.r.t. ``t``,

    and additionally first- and second-order derivatives of the solution.

    The derivatives w.r.t.``y0, p, q`` are computed based on the IND and automatic differentiation (AD)
    principles. Hence, their computation is accurate (close to machine precision and efficient.

    Both the forward and reverse/adjoint mode computations are supported.


Rationale:

    * For optimal control (direct approach) one requires accurate derivatives of the solution w.r.t. controls
    ``q``.

    * For least-squares parameter estimation algorithms one requires derivatives of the solution w.r.t. parameters ``p``.

    * For experimental design optimization one requires accurate second-order derivatives of the solution w.r.t. ``p`` and ``q``

Known to work on:

    * Ubuntu 12.04, Tapenade 3.6


Backend:

    The integration algorithms are written in Python and repeatedly evaluate the rhs, i.e.,
     ``f(y, p, q)`` and its derivatives ``df/d(y,p,q) (y, p, q)``,

    INDegrator currently only supports model functions are written in Fortran 77 and differentiates them
    using the AD tool Tapenade. This approach yields very efficient code.


Requirements:

    You need Tapenade >= 3.6 to generate the derivatives of the model functions.



Getting started:
    
    Run ``python bimolkat.py``  


Features:

    * Explicit Euler, fixed stepsize
     - first-order forward
     - first-order reverse



-------------------------------------------------------------------------------

Licence: GPL v3

    INDegrator, an internal numerical differentiation (IND) library of differentiated ODE solvers
    Copyright (C) 2014  Sebastian F. Walter

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.