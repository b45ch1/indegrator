from cffi import FFI
import os

ffi = FFI()
ffi.cdef("""
    void ffcn_(double *t, double *x, double *f, double *p, double *u);
""")



class libproblem(object):

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
