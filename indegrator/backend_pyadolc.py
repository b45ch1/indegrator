import os
import numpy
import sys
import adolc

class BackendPyadolc(object):

    def __init__(self, ffcn):
        """
        is ffcn either

        1)  a callable  of the form ``def ffcn(t, x, f, p, u)``
        2) or a path to a directory containing a file ffcn.py

        """


        if hasattr(ffcn, '__call__'):
            self.ffcn = ffcn
        else:
            self.path = os.path.abspath(ffcn)
            self.dir  = os.path.dirname(self.path)
            
            sys.path.insert(0, self.dir)
            import ffcn

            self.ffcn = ffcn.ffcn


        self.traced = False


    def trace(self, dims):
        # trace function
        t = numpy.zeros(1)
        x = numpy.zeros(dims['x'])
        f = numpy.zeros(dims['x'])
        p = numpy.zeros(dims['p'])
        u = numpy.zeros(dims['u'])

        adolc.trace_on(123)
        at = adolc.adouble(t)
        ax = adolc.adouble(x)
        af = adolc.adouble(f)
        ap = adolc.adouble(p)
        au = adolc.adouble(u)

        adolc.independent(at)
        adolc.independent(ax)
        adolc.independent(ap)
        adolc.independent(au)
        self.ffcn(t, ax, af, ap, au)
        adolc.dependent(af)
        adolc.trace_off()
        self.traced = True

    def txpu_to_v(self, t, x, p, u):
        v = numpy.zeros(t.size + x.size + p.size + u.size)
        v[0:1]                      = t
        v[1:x.size+1]               = x
        v[1+x.size:1+x.size+p.size] = p
        v[1+x.size+p.size:]         = u
        return v

    def dot_txpu_to_v(self, t, t_dot, x, x_dot, p, p_dot, u, u_dot):
        nbdirs = t_dot.shape[1]
        v = numpy.zeros((t.size + x.size + p.size + u.size, nbdirs))
        v[0:1]                         = t_dot
        v[1:x.size+1, :]               = x_dot
        v[1+x.size:1+x.size+p.size, :] = p_dot
        v[1+x.size+p.size:, :]         = u_dot
        return v

    def bar_v_to_txpu(self, v, v_bar, x, x_bar, p, p_bar, u, u_bar):
        # t_bar  += v_bar[0:1]
        x_bar  += v_bar[1:x.size+1]
        p_bar  += v_bar[1+x.size:1+x.size+p.size]
        u_bar  += v_bar[1+x.size+p.size:]

    def ffcn(self, t, x, f, p, u):
        if self.traced == False:
            dims = {'x': x.size, 'p': p.size, 'u': u.size}
            self.trace(dims)
        v = self.txpu_to_v(t,x,p,u)
        f[:] = adolc.function(123, v)

    def ffcn_dot(self, t, x, x_dot, f, f_dot, p, p_dot, u, u_dot):
        if self.traced == False:
            dims = {'x': x.size, 'p': p.size, 'u': u.size}
            self.trace(dims)

        v = self.txpu_to_v(t,x,p,u)
        t_dot = numpy.zeros((1, x_dot.shape[1]))
        v_dot = self.dot_txpu_to_v(t, t_dot, x, x_dot, p, p_dot, u, u_dot)

        f[:], f_dot[:,:] = adolc.fov_forward(123, v, v_dot)

    def ffcn_bar(self, t, x, x_bar, f, f_bar, p, p_bar, u, u_bar):
        if self.traced == False:
            dims = {'x': x.size, 'p': p.size, 'u': u.size}
            self.trace(dims)

        v = self.txpu_to_v(t,x,p,u)
        adolc.zos_forward(123, v, 1)
        v_bar = adolc.fos_reverse(123, f_bar)
        self.bar_v_to_txpu(v, v_bar, x, x_bar, p, p_bar, u, u_bar)

    def ffcn_ddot(self, t,
                  x, x_dot2, x_dot1, x_ddot, 
                  f, f_dot2, f_dot1, f_ddot,
                  p, p_dot2, p_dot1, p_ddot,
                  u, u_dot2, u_dot1, u_ddot):

        if self.traced == False:
            dims = {'x': x.size, 'p': p.size, 'u': u.size}
            self.trace(dims)

        N = t.size + x.size + p.size + u.size

        M = x_dot1.shape[1]
        L = x_dot2.shape[1]

        v   = numpy.zeros(N) 
        V1  = numpy.zeros((N, x_dot1.shape[1]))
        V2  = numpy.zeros((N, x_dot2.shape[1]))
        V12 = numpy.zeros((N, x_ddot.shape[1], x_ddot.shape[2]))
        V   = numpy.zeros((N, M + L + M*L, 2))

        v[1:1+x.size]                       = x
        v[1+x.size:1+x.size+p.size]         = p
        v[1+x.size+p.size:]                 = u

        V1[1:1+x.size, :]                   = x_dot1
        V1[1+x.size:1+x.size+p.size, :]     = p_dot1
        V1[1+x.size+p.size:, :]             = u_dot1

        V2[1:1+x.size, :]                   = x_dot2
        V2[1+x.size:1+x.size+p.size, :]     = p_dot2
        V2[1+x.size+p.size:, :]             = u_dot2

        V12[1:1+x.size, :, :]               = x_ddot
        V12[1+x.size:1+x.size+p.size, :, :] = p_ddot
        V12[1+x.size+p.size:, :, :]         = u_ddot

        adolc.interpolation.entangle_cross(V, V1, V2, V12)
        w, W = adolc.hov_forward(123, v, V)

        U1  = numpy.zeros((x.size, V1.shape[1]))
        U2  = numpy.zeros((x.size, V2.shape[1]))
        U12 = numpy.zeros((x.size, M, L))

        f[:] = w
        adolc.interpolation.detangle_cross(W, U1, U2, U12)
        f_dot1[:, :]    = U1
        f_dot2[:, :]    = U2
        f_ddot[:, :, :] = U12
