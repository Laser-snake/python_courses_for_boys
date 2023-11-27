import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel import prange, parallel
from libc.math cimport sqrt, cos, sin


def multip_func(np_arr_z, np_arr_y):
    result = np.zeros((np_arr_z.shape[0], np_arr_y.shape[1]))
    cdef double[:, :] np_arr_mv_z = np_arr_z
    cdef double[:, :] np_arr_mv_y = np_arr_y
    cdef double[:, :] result_mv = result
    if (1 == 1):
        func1(np_arr_mv_z, np_arr_mv_y, result_mv)
        return result#func1(np_arr_mv_z, np_arr_mv_y, result_mv)
    else:
        return "Не верный формат ввода"


#@cython.boundscheck(False)
#@cython.wraparound(False)
#@cython.initializedcheck(False)
#@cython.cdivision(True)
cdef void func1(double[:, :] z_mv, double[:, :] y_mv,double[:, :] result_mv):
    cdef int i, j, k
    #cdef double sum = 0.0
    #cdef double s = 0.0
    with nogil:
        for i in prange(z_mv.shape[0], num_threads=2):
            for j in range(y_mv.shape[1]):
                #sum = 0.0
                #k = 0
                for k in range(z_mv.shape[1]):
                     result_mv[i, j] += z_mv[i, k] * y_mv[k, j]

 #   return result_mv



