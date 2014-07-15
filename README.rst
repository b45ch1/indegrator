INDegrator, an internal numerical differentiation (IND) library of differentiated ODE solvers
---------------------------------------------------------------------------------------------


Description:

    INDegrator is a library of IND integration schemes.

    They allow you to evaluate the solution ``y(t; y0, p, q)`` of initial value
    problem (IVP) of the form::


        y_t  = f(y, p, q)
        y(0) = y0

    where y_t denotes the derivative of ``y`` w.r.t. ``t``,

    and additionally 

    * first-order derivatives ``y_y0(t)``, ``y_p(t)``, ``y_q(t)``, 
    * and second-order derivatives of the solution

    in an accurate and efficient way.

    The derivatives w.r.t.``y0, p, q`` are computed based on the IND and automatic differentiation (AD)
    principles. Both forward and reverse/adjoint mode computations are supported.

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
     - second-order forward
     - first-order reverse



-------------------------------------------------------------------------------

Licence:
    BSD style using http://www.opensource.org/licenses/bsd-license.php template
    as it was on 2009-01-24 with the following substutions:

    * <YEAR> = 2008-2009
    * <OWNER> = Sebastian F. Walter, sebastian.walter@gmail.com
    * <ORGANIZATION> = Heidelberg University
    * In addition, "Neither the name of the contributors' organizations" was changed to "Neither the names of the contributors' organizations"


Copyright (c) 2014 Seastian F. Walter
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.
    * Neither the names of the contributors' organizations nor the names of
      its contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.