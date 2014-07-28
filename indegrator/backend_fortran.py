from cffi import FFI
import os
import numpy

ffi = FFI()
ffi.cdef("""
    void ffcn_(double *t, double *x, double *f, double *p, double *u);

    void ffcn_d_xpu_v_(double *t, double *x, double *x_dot,
                                  double *f, double *f_dot,
                                  double *p, double *p_dot,
                                  double *u, double *u_dot,
                                  int *nbdirs);

    void ffcn_b_xpu_(double *t, double *x, double *x_bar,
                               double *f, double *f_bar,
                               double *p, double *p_bar,
                               double *u, double *u_bar);

    void ffcn_d_xpu_v_d_xpu_v_(double *t,
                               double *x, double *x_dot1, double *x_dot2, double *x_ddot,
                               double *f, double *f_dot1, double *f_dot2, double *f_ddot,
                               double *p, double *p_dot1, double *p_dot2, double *p_ddot,
                               double *u, double *u_dot1, double *u_dot2, double *u_ddot,
                               int *nbdirs2, int *nbdirs1);

""")


class BackendFortran(object):

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.dir  = os.path.dirname(self.path)
        # Load symbols from the current process (Python).
        self.lib = ffi.dlopen(self.path)
        print('Loaded lib {0}'.format(self.lib))

    def ffcn(self, t, x, f, p, u):

        ffi_t = ffi.cast("double *", t.ctypes.data)
        ffi_x = ffi.cast("double *", x.ctypes.data)
        ffi_f = ffi.cast("double *", f.ctypes.data)
        ffi_p = ffi.cast("double *", p.ctypes.data)
        ffi_u = ffi.cast("double *", u.ctypes.data)

        self.lib.ffcn_(ffi_t, ffi_x, ffi_f, ffi_p, ffi_u)

    def ffcn_dot(self, t, x, x_dot, f, f_dot, p, p_dot, u, u_dot):

        nbdirs = numpy.array([x_dot.shape[1]], dtype=numpy.int32)

        ffi_t = ffi.cast("double *", t.ctypes.data)
        ffi_x = ffi.cast("double *", x.ctypes.data)
        ffi_f = ffi.cast("double *", f.ctypes.data)
        ffi_p = ffi.cast("double *", p.ctypes.data)
        ffi_u = ffi.cast("double *", u.ctypes.data)

        ffi_x_dot = ffi.cast("double *", x_dot.ctypes.data)
        ffi_f_dot = ffi.cast("double *", f_dot.ctypes.data)
        ffi_p_dot = ffi.cast("double *", p_dot.ctypes.data)
        ffi_u_dot = ffi.cast("double *", u_dot.ctypes.data)

        ffi_nbdirs = ffi.cast("int *", nbdirs.ctypes.data)

        self.lib.ffcn_d_xpu_v_(ffi_t,
                               ffi_x, ffi_x_dot,
                               ffi_f, ffi_f_dot,
                               ffi_p, ffi_p_dot,
                               ffi_u, ffi_u_dot,
                               ffi_nbdirs)

    def ffcn_bar(self, t, x, x_bar, f, f_bar, p, p_bar, u, u_bar):

        ffi_t = ffi.cast("double *", t.ctypes.data)
        ffi_x = ffi.cast("double *", x.ctypes.data)
        ffi_f = ffi.cast("double *", f.ctypes.data)
        ffi_p = ffi.cast("double *", p.ctypes.data)
        ffi_u = ffi.cast("double *", u.ctypes.data)

        ffi_x_bar = ffi.cast("double *", x_bar.ctypes.data)
        ffi_f_bar = ffi.cast("double *", f_bar.ctypes.data)
        ffi_p_bar = ffi.cast("double *", p_bar.ctypes.data)
        ffi_u_bar = ffi.cast("double *", u_bar.ctypes.data)

        self.lib.ffcn_b_xpu_(ffi_t,
                             ffi_x, ffi_x_bar,
                             ffi_f, ffi_f_bar,
                             ffi_p, ffi_p_bar,
                             ffi_u, ffi_u_bar)

    def ffcn_ddot(self, t,
                  x, x_dot2, x_dot1, x_ddot, 
                  f, f_dot2, f_dot1, f_ddot,
                  p, p_dot2, p_dot1, p_ddot,
                  u, u_dot2, u_dot1, u_ddot):

        nbdirs1 = numpy.array([x_dot1.shape[1]], dtype=numpy.int32)
        nbdirs2 = numpy.array([x_dot2.shape[1]], dtype=numpy.int32)

        ffi_t = ffi.cast("double *", t.ctypes.data)
        ffi_x = ffi.cast("double *", x.ctypes.data)
        ffi_f = ffi.cast("double *", f.ctypes.data)
        ffi_p = ffi.cast("double *", p.ctypes.data)
        ffi_u = ffi.cast("double *", u.ctypes.data)

        ffi_x_dot1 = ffi.cast("double *", x_dot1.ctypes.data)
        ffi_f_dot1 = ffi.cast("double *", f_dot1.ctypes.data)
        ffi_p_dot1 = ffi.cast("double *", p_dot1.ctypes.data)
        ffi_u_dot1 = ffi.cast("double *", u_dot1.ctypes.data)

        ffi_x_dot2 = ffi.cast("double *", x_dot2.ctypes.data)
        ffi_f_dot2 = ffi.cast("double *", f_dot2.ctypes.data)
        ffi_p_dot2 = ffi.cast("double *", p_dot2.ctypes.data)
        ffi_u_dot2 = ffi.cast("double *", u_dot2.ctypes.data)

        ffi_x_ddot = ffi.cast("double *", x_ddot.ctypes.data)
        ffi_f_ddot = ffi.cast("double *", f_ddot.ctypes.data)
        ffi_p_ddot = ffi.cast("double *", p_ddot.ctypes.data)
        ffi_u_ddot = ffi.cast("double *", u_ddot.ctypes.data)

        ffi_nbdirs1 = ffi.cast("int *", nbdirs1.ctypes.data)
        ffi_nbdirs2 = ffi.cast("int *", nbdirs2.ctypes.data)

        self.lib.ffcn_d_xpu_v_d_xpu_v_(ffi_t,
                               ffi_x, ffi_x_dot2, ffi_x_dot1, ffi_x_ddot,
                               ffi_f, ffi_f_dot2, ffi_f_dot1, ffi_f_ddot,
                               ffi_p, ffi_p_dot2, ffi_p_dot1, ffi_p_ddot,
                               ffi_u, ffi_u_dot2, ffi_u_dot1, ffi_u_ddot,
                               ffi_nbdirs1, ffi_nbdirs2)