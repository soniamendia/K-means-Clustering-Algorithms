from random import random 
def pi_montecarlo(n=1000)
cdef int n_in = 0, 1
cdef double x, y
    n_in = 0
    for i in range(n):
        x,y = random(), random()

        if x**2 + y**2 <= 1:
            n_in+= 1
    return  4.0 * n_in / n
    