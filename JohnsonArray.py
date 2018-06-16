# Make array of Johnson graph independence numbers

import numpy as np
import random
import networkx as nx
import networkx.algorithms.approximation as nxaa
import matplotlib.pyplot as plt
import networkx.generators.directed
import itertools as it
import time as tm
from GraphFun import *

dtype = "int32"
niters = 100 # Number of iterations
ntrials = 40 # Number of times to repeat algorithm
max_v = 21

# Compute independence array
alphas = np.zeros((max_v,max_v),dtype=dtype)
for v in range(max_v):
    for k in range(v//2+1):
        G = genJohnsonGraph(v,k,k-1)
        A = getAdjArray(G)
        maxIndSet = fastFindIndSet(A,niters,ntrials)
        alpha = np.sum(maxIndSet)
        alphas[v,k] = alpha
        alphas[v,v-k] = alpha
        np.savetxt("JohnsonAlphas.txt",alphas,fmt='%5d')
