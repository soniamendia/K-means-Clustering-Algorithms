from random import random 
def pi_montecarlo(n=1000)
    n_in = 0
    for i in range(n):
        x,y = random(), random()

        if x**2 + y**2 <= 1:
            n_in+= 1
    return  4 * n_in / n


