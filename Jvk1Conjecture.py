# Finding independent sets with Boltzmann machines

import numpy as np
import time as tm
from GraphFun import *
from scipy.special import binom

dtype = "int32"
niters = 80 #100000# Number of iterations
ntrials = 50 # Number of times to repeat algorithm

for k in range(2,20):
    print('\n')
    print('k =',k)
    correct = []
    tic = tm.time()
    V,A = genJohnsonAdjList(3*k-3,k,1)
    best_set = fastFindIndSetAlt(A,niters,ntrials)
    beta = np.sum(best_set)
    gamma = binom(3*k-5,k-2)
    toc = tm.time()
    if beta<=gamma:
        print("Success!")
    else:
        print("Fail")
