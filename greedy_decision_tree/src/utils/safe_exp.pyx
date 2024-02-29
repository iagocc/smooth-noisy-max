from libc.math cimport expl
from libc.stdio cimport printf
cimport numpy as cnp

cpdef cnp.npy_float128 safe_exp(cnp.npy_float128 val):
    printf("(CYT) %Lf - %Lf\n", val, expl(val))

    return expl(val)
