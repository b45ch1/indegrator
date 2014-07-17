

=============================================================================================
INDegrator, an internal numerical differentiation (IND) library of differentiated ODE solvers
=============================================================================================


Description
-----------

    INDegrator is a library of IND integration schemes.

    They allow you to evaluate the solution :math:`y(t; y_0, p, q)` of initial value
    problems (IVP) of the form


    .. math::

        y_t =& f(y, p, q) \\
        y(0) =& y_0

    where y_t denotes the derivative of ``y`` w.r.t. ``t``,

    and additionally 

    * first-order derivatives 

      .. math::

        \frac{\partial y}{\partial y_0}(t; y_0, p, q) \;, \\
        \frac{\partial y}{\partial p}(t; y_0, p, q) \;, \\
        \frac{\partial y}{\partial q}(t; y_0, p, q) 

    * and second-order derivatives of the solution

      .. math::

        \frac{\partial^2 y}{\partial y_0^2}(t; y_0, p, q) \;, &
        \frac{\partial^2 y}{\partial y_0 \partial p}(t; y_0, p, q) \\
        \frac{\partial^2 y}{\partial y_0 \partial q}(t; y_0, p, q) \;, &
        \frac{\partial^2 y}{\partial p^2}(t; y_0, p, q) \\
        \frac{\partial^2 y}{\partial p \partial q}(t; y_0, p, q) \;, &
        \frac{\partial^2 y}{\partial q^2}(t; y_0, p, q) 




    in an accurate and efficient way.

    The derivatives w.r.t. :math:`y_0`, :math:`p` and :math:`q` are computed based on the IND and automatic differentiation (AD)
    principles. Both forward and reverse/adjoint mode computations are supported.

Rationale
---------

    * For optimal control (direct approach) one requires accurate derivatives of the solution w.r.t. controls ``q``.

    * For least-squares parameter estimation algorithms one requires derivatives of the solution w.r.t. parameters ``p``.

    * For experimental design optimization one requires accurate second-order derivatives of the solution w.r.t. ``p`` and ``q``


Features
--------

    * Explicit Euler, fixed stepsize

         - first-order forward
         - second-order forward
         - first-order reverse

    * Runge Kutta 4 (RK4), fixed stepsize

         - first-order forward
         - second-order forward
         - first-order reverse


Known to work on
----------------

    * Ubuntu 12.04, Tapenade 3.6


Backend
-------

    The integration algorithms are written in Python and repeatedly evaluate the rhs, i.e.,
     ``f(y, p, q)`` and its derivatives ``df/d(y,p,q) (y, p, q)``,

    INDegrator currently only supports model functions are written in Fortran 77 and differentiates them
    using the AD tool Tapenade. This approach yields very efficient code.


Requirements
------------

    You need Tapenade >= 3.6 to generate the derivatives of the model functions.
    Get it on http://www-sop.inria.fr/tropics/tapenade.html


Getting started
---------------
    
    

Example 1: zero order forward
`````````````````````````````
    
    Compute trajectory :math:`y(t; y_0, p, q)`.

    .. literalinclude:: bimolkat_zo_forward.py
        :lines: 1-30

.. image:: bimolkat_zo_forward.png
    :align: center
    :scale: 100



Example 2: zero order forward
`````````````````````````````

    Compute trajectory :math:`\frac{\partial y}{\partial p}(t; y_0, p, q)`
    and :math:`\frac{\partial y}{\partial q}(t; y_0, p, q)`

    .. literalinclude:: bimolkat_fo_forward.py

  .. image:: bimolkat_fo_forward_p.png
    :align: center
    :scale: 100

  .. image:: bimolkat_fo_forward_q.png
    :align: center
    :scale: 100



Example 3: zero order forward
`````````````````````````````

    Compute the gradients of the state :math:`y(t=2; y_0, p, q)` w.r.t. :math:`y_0, p, q`, i.e.,

    .. math::

        \nabla_{y_0} y(t=2; y_0, p, q) \;, \\
        \nabla_{p} y(t=2; y_0, p, q) \;, \\
        \nabla_{q} y(t=2; y_0, p, q) \;.


    .. literalinclude:: bimolkat_fo_reverse.py

    where one obtains the output::

        gradient of x(t=2; x0, p, q) w.r.t. p  = 
        [-0.05553826 -0.26378935 -0.18022685 -0.57068079  0.0912674 ]
        gradient of x(t=2; x0, p, q) w.r.t. q  = 
        [[[ -5.88595438e-08   0.00000000e+00]
          [ -5.93303903e-08   0.00000000e+00]
          [ -5.98124990e-08   0.00000000e+00]
          ..., 
          [ -1.55170436e-04   0.00000000e+00]
          [ -1.57724384e-04   0.00000000e+00]
          [  0.00000000e+00   0.00000000e+00]]

         [[ -1.91875726e-04   0.00000000e+00]
          [ -1.92256884e-04   0.00000000e+00]
          [ -1.92638781e-04   0.00000000e+00]
          ..., 
          [ -1.09278673e-05   0.00000000e+00]
          [ -3.67184216e-06   0.00000000e+00]
          [  0.00000000e+00   0.00000000e+00]]

         [[ -1.88832838e-03   0.00000000e+00]
          [ -1.88832379e-03   0.00000000e+00]
          [ -1.88831907e-03   0.00000000e+00]
          ..., 
          [ -4.59399550e-05   0.00000000e+00]
          [ -1.54234950e-05   0.00000000e+00]
          [  0.00000000e+00   0.00000000e+00]]

         [[  2.08733332e-03   0.00000000e+00]
          [  2.08733821e-03   0.00000000e+00]
          [  2.08734323e-03   0.00000000e+00]
          ..., 
          [  3.96132167e-03   0.00000000e+00]
          [  3.99233933e-03   0.00000000e+00]
          [  0.00000000e+00   0.00000000e+00]]]
        gradient of x(t=2; x0, p, q) w.r.t. x0 = 
        [-0.47113849  0.52078906  0.04958182  0.0499501  -0.04782551]


