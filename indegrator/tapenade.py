import os
import re
import glob
import string
import shutil
import subprocess


import string, re, os, shutil

def trim_trailing_whitespace(s):
    t = ''
    for line in s.splitlines():
         t += line.rstrip() + os.linesep
    return t

def remove_exclamation_mark_comments(s):
    """
    removes comments of the type::

        ``some code ! some comment``

    """
    # remove ! comments when followed by line continuation
    s = re.sub(r"!.*", "", s)
    s = trim_trailing_whitespace(s)
    return s


def do_fortran_line_continuation(s, max_cols = 72):
    """
    out = do_fortran_line_continuation(s, max_cols = 72)

    Splits lines with more than max_cols = 72 characters.

    Parameters:

    :s: string,
        Fortran code

    Returns:

    :out: string,
        modified Fortran code

    """

    s = remove_exclamation_mark_comments(s)
    lines = s.splitlines()
    s = ""

    for line in lines:
        while len( line ) > max_cols and line[0] == ' ':
           s += line[:max_cols] + os.linesep
           line = "     *" + line[max_cols:]
        s += line + os.linesep

    return s

def undo_fortran_line_continuation(s):
    """

    out = undo_fortran_line_continuation(s)

    Write code spread over several lines into one line

    Parameters:

    :s: string,
        Fortran code

    Returns:

    :out: string,
        modified Fortran code

    """
    # return re.sub(r"(\r)?\n     [*+\S]", "", s)

    return re.sub(r"(\r)?\n     [*+\S]", "", s)


def find_and_remove_lines(text, search_string):
    """

    out = find_and_remove_lines(text, search_string)

    Deletes all lines in `text` that contain `search_string`
    """
    s = ""
    for row in text.splitlines():
        if row.find(search_string) == -1:
            s += row + os.linesep

    return s

def read_file(path, line_continuation = False):
    f = open( path, "r" )
    out = f.read()
    f.close()

    if line_continuation:
        out = undo_fortran_line_continuation(out)

    return out

def write_file(path, s, line_continuation = False):
    if line_continuation:
        s = do_fortran_line_continuation(s)

    f = open( path, "w" )
    f.write(s)
    f.close()



def change_tapenade_forward_generated_files(path, retvals, replace_nbdirxmax = True):
    """ set variables that tapenade requires and couldn't deduce from the file

    Parameters:

    :retvals:   list of strings,
                The return value of a fucntion, e.g. in ffcn(t,x,f,p,q) it would be ['f']
    """

    # read file
    s = read_file(path, line_continuation=True)

    # change string
    s = change_tapenade_forward_generated_code(s, retvals,  replace_nbdirxmax = replace_nbdirxmax)

    # write file
    write_file(path, s, line_continuation=True)



def change_tapenade_forward_generated_code(s, retvals, replace_nbdirxmax = True):
    """

    s = change_tapenade_forward_generated_code(s, retval, dims, replace_nbdirxmax = True)

    Parameters:

    :s:         string
                source code to be modified

    :retvals:   list of strings
                return value, e.g. 'f' for ffcn or 'g' for gfcn

    :dims:      dict
                dimensions of the variables

    :replace_nbdirxmax:  bool
                replace nbdirsmax with nbdirs?

    Returns:

    :s:         string
                modified source code

    """

    def unique(items):
        """ helper function to find and remove duplicate lines """
        found = set([])
        keep = []

        for item in items:
            if item in found and re.match(r'\s*INTEGER.*', item, flags=re.IGNORECASE) != None:
                continue
            found.add(item)
            keep.append(item)

        return keep

    # all hints of the type
    # C  Hint: ISIZE1OFmr_d_bINmess3_d_xp_v should be the value of nbdirs
    results = re.findall('C  Hint: (\w+?) should be the value of (\w+)', s, re.IGNORECASE)
    for result in results:
        # remove hint
        s = re.sub('C  Hint: %s should be the value of %s.*\n'%result, "", s)
        # apply hint
        s = re.sub( result[0], result[1], s)

    # first all Hints concerning ISIZE20F
    results = re.findall('C  Hint: ISIZE2OF(\w+?) should be the value of (\w+)', s, re.IGNORECASE)
    for result in results:
        # remove hint
        s = re.sub('C  Hint: ISIZE2OF%s should be the value of %s.*\n'%result, "", s)
        # apply hint
        s = re.sub( 'ISIZE2OF' + result[0], result[1], s)


    # assume that the first element of retvals is the appropriate dimension
    # for the source transformation

    # find all subroutines in s and treat them separately
    subroutines = re.findall(r'\s*SUBROUTINE.*?\n\s*END[\s$](?!IF|DO)', s, re.S | re.IGNORECASE)
    tmp = ''
    for s in subroutines:

        if replace_nbdirxmax:
            # replace nbdirsmax with the last argument in subroutine ffcn(..,..,nbdirs)
            nbdirsmax_replacement = re.match(r'\s*SUBROUTINE.*?,\s*(?P<nbdirs>\w+)\s*\)', s, re.IGNORECASE).group('nbdirs')
            s = s.replace("nbdirsmax", nbdirsmax_replacement)
            s = re.sub('C  Hint: \w+ should be the maximum number of differentiation directions.*\n', "", s)


        # in the case when the function is empty, tapenade adds type declarations twice,
        # e.g. INTEGER ii1  INTEGER ii1
        # find and remove those duplicates
        s = string.join( unique(s.split(os.linesep)), os.linesep)
        tmp += s

    s = tmp

    # by default: do what Tapenade suggests in the Hints
    # ##################################################

    # find calls to external subroutines which are defined in the same file
    subs = re.findall(r'\s*SUBROUTINE\s+(\w+?)\s*\(', s, flags=re.IGNORECASE)
    if len(subs)>0:
        top_function = subs[0].replace('_','')

        ext_subs = re.findall(r'\s*CALL\s+(\w+)\s*\(', s, flags=re.IGNORECASE)
        ext_subs_with_declaration = []
        for sub in ext_subs:
            # look for function declaration
            if len(re.findall(r'\s*SUBROUTINE\s+%s\s*\('%sub, s, flags=re.IGNORECASE))>0:
                ext_subs_with_declaration.append(sub)

        # and rename these functions
        for sub in ext_subs_with_declaration:
            pattern = re.compile(r'(\s*SUBROUTINE\s*|\s*CALL\s*)(%s)'%sub,
                      flags=re.IGNORECASE)
            s = re.sub(pattern, r'\1%s'%(top_function+'_'+sub), s)

        # print 'ext_subs',ext_subs
        # print 'ext_subs_with_declaration', ext_subs_with_declaration

    s = find_and_remove_lines(s, "INCLUDE 'DIFFSIZES.inc'")


    return s



def tapenade_params(mode, path, functionname, filesuffix):
    """
    build tapenade parameters

    mode:       string
                'forward', 'forward_sd', 'reverse'
    path:       string or list of strings
                paths to input files
    filesuffix: string
                append string, e.g. _d_{filesuffix}


    path are the files that tapenade considers during differentiation.
    When there are external subroutines, then paths is either a list ['ffcn.f', 'myaux.f']
    or a string 'ffcn.f myaux.f'.
    """

    path=path.split()

    if isinstance(path, list):
        directory = os.path.dirname(path[0])
        path = string.join(path)
    else:
        directory = os.path.dirname(path)


    if mode == 'forward' or mode == 'forward_sd':
        method = '-forward'
        diffvarname = "_d" # appended to variable names

        if mode == 'forward':
            multi = '-multi'
            difffuncname="_d_" + filesuffix + "_"
            difffun = functionname + difffuncname + "v"

        elif mode == 'forward_sd':
            # print 'forward_sd mode'
            multi = ''
            difffuncname="_d_" + filesuffix
            difffun = functionname + difffuncname

        output_path = os.path.join(directory, difffun + ".f")
        optim = '-nooptim "deadcontrol" '


    elif mode == 'reverse':
        method = '-reverse'
        multi  = ''
        diffvarname = "_b"                       # appended to variable names
        difffuncname = "_"+"b"+"_"+filesuffix   # appended to function name
        difffun = functionname+ difffuncname
        output_path = os.path.join(directory, difffun + ".f")
        optim = '-nooptim "deadcontrol" -nooptim "adjointliveness"  '

    else:
        raise ValueError('mode has to be "forward" or "reverse", \
                         but you provided "%s"'%mode)

    # print directory
    # print output_path
    # raw_input('enter')


    return method, multi, optim, difffun, difffuncname, diffvarname, output_path



def call_tapenade(mode, path, functionname, x, y, filesuffix):
    """
    differentiates function `functionname` in file `filename` w.r.t. to the
    variables `x`

    If the output file already exists, the differentiation is skipped.


    calls TAPENADE in the forward mode
    the differentiated files lie in the directory `self.outputdir`

    INPUT:
        mode                string      'forward', 'forward_sd' or 'reverse'
                                        'forward' is by default the
                                        multidirectional tangent propagation
                                        mode whereas 'forward_sd' propagates
                                        just one tangent direction

        path                string      paths to the file,
                                        e.g. `~/vplan/examples/A/fortran/ffcn.f ~/vplan/examples/A/fortran/myaux.f`

        function_name       string      function name to be differentiated,
                                        e.g. ffcn

        x                   dict        independent variables with dimension,
                                        e.g. x = ['x1','x2']

        y                   dict        y = fun(x), i.e. dependent variables,
                                        e.g. y = 'y'

        filesuffix          string      string that appears in the output
                                        filename

    OUTPUT:
        output_path         string      name of the function as it appears in
                                        the differentiated code

        diff_function_name  string      name of the diff. subroutine


     EXAMPLE:
     tapenade_call('forward', /tmp/inputs/ffcn.f','ffcn, ['x1','x2'], 'y', 'A')

     returns output_path = '/tmp/inputs/ffcn_d_A_v.f'

    """

    if isinstance(path, list):
        directory = os.path.dirname(path[0])
        path = string.join(path)
    else:
        directory = os.path.dirname(path)

    if isinstance(x, list):
        independent_vars = string.join(x)
    else:
        independent_vars = x

    if isinstance(y, list):
        dependent_vars = string.join(y)
    else:
        dependent_vars = y

    # print 'directory=', directory, 'Differentiating:' + path


    method, multi, optim, difffun, difffuncname, diffvarname, output_path = tapenade_params(mode, path, functionname, filesuffix)

    # print output_path
    # raw_input('enter')

    command = 'tapenade %s -fixinterface ' \
              '%s %s ' \
              '-inputlanguage "fortran" -outputlanguage "fortran" ' \
              '-root "%s" -vars "%s" -outvars "%s" ' \
              '-output "%s" -difffuncname "%s" -diffvarname "%s" ' \
              '\%s -O "%s"   '\
              % ( method, optim, multi, functionname, independent_vars, dependent_vars, functionname,
                  difffuncname, diffvarname, path, os.path.dirname(output_path) )


    if not os.path.exists(output_path):
        # only call tapenade if the file doesn't exist
        os.popen(command).close()
    # else:
    #     print("%s already exists: skipping"%functionname)

    # write command to a file
    cmd_file_path = os.path.join(directory, difffun + "_tapenade_cmd.txt")
    cmd_file = open(cmd_file_path, 'w')
    cmd_file.write('%s\n'%command)
    cmd_file.close()

    # check that the desired file was created
    if not os.path.exists(output_path):
        err_str = 'Some error occurred when calling tapenade on the function %s\n'%output_path
        err_str+= 'Tapenade reported the error:\n```\n'
        with open (os.path.join(directory, difffun + '.msg'), "r") as myfile:
            err_str += myfile.read()
        err_str+= '\n```'
        raise Exception(err_str)

    # create backup file for introspection
    shutil.copy(output_path, output_path + ".bak")

    return output_path, difffun


class Differentiator(object):

    Makefile = \
'''

LIB        = so
OPT        = -O3 
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
'''

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.dir  = os.path.dirname(self.path)

        print self.path
        print self.dir
        self.clean()


        # first order
        call_tapenade('forward', self.path, 'ffcn', ['x', 'p', 'u'], ['f'], 'xpu')
        change_tapenade_forward_generated_files(os.path.join(self.dir, 'ffcn_d_xpu_v.f'), ['f'], replace_nbdirxmax = True)

        call_tapenade('reverse', self.path, 'ffcn', ['x', 'p', 'u'], ['f', 'x', 'p', 'u'], 'xpu')
        change_tapenade_forward_generated_files(os.path.join(self.dir, 'ffcn_b_xpu.f'), ['f'], replace_nbdirxmax = True)

        # second order
        path2 = os.path.join(self.dir, 'ffcn_d_xpu_v.f')
        call_tapenade('forward', path2, 'ffcn_d_xpu_v', ['x', 'x_d', 'p', 'p_d', 'u', 'u_d'], ['f', 'f_d'], 'xpu')
        change_tapenade_forward_generated_files(os.path.join(self.dir, 'ffcn_d_xpu_v_d_xpu_v.f'), ['f'], replace_nbdirxmax = True)


        self.make()
     

    def clean(self):
        files = glob.glob(os.path.join(self.dir, 'ffcn[^_]*'))
        files +=  glob.glob(os.path.join(self.dir, '~'))
        files +=  glob.glob(os.path.join(self.dir, '.bak'))
        files +=  glob.glob(os.path.join(self.dir, 'libproblem.*'))
        for f in files:
            os.remove(f)

        for f in glob.glob(os.path.join(self.dir, '*.o')): os.remove(f)


    def make(self):

        with open( os.path.join(self.dir, 'Makefile'), "w" ) as f:
            f.write(self.Makefile)

        cwd = os.getcwd()
        os.chdir(self.dir)
        p = subprocess.check_output('make', stderr=subprocess.STDOUT)
        os.chdir(cwd)

