

LIB        = so
OPT        = -O0 -g
MAKELIB    = g++ -shared -o
COMPILE_C  = gcc      -c -o $@ $(OPT) -fPIC
COMPILE_F  = gfortran -c -o $@ $(OPT) -fPIC -fno-second-underscore -frecursive -finit-local-zero -fdefault-real-8 -fdefault-double-8


%.o : %.c
	$(COMPILE_C) -fPIC $<

%.o : %.f
	$(COMPILE_F) $<

CSRCS=$(wildcard *.c)
COBJS=$(CSRCS:.c=.o)

FSRCS=$(wildcard *.f)
FOBJS=$(FSRCS:.f=.o)

all: $(FOBJS) $(COBJS)
	$(MAKELIB) libproblem.$(LIB) *.o -lgfortran
