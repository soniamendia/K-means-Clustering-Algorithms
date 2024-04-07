# cython profileTrue
import numpy as np

def numpy_py():
    py_arr = np.random.rand(1000)
    for i in range(1000):
        py_arr[i] +=1

def numpy_c():
    cdef3 cnp.ndarray[cnp.int_t, ndim = 1] c_arr
    c_arr = np.random.rand(1000)
    cdef int 1

    for i in range(1000):
        c_array[1] += 1


