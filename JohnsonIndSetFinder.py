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
ntrials = 100 # Number of times to repeat algorithm
max_v = 21

## If starting from scratch (overwrites arrays if uncommented!)
#alphas = -np.ones((max_v,max_v),dtype=dtype)
#maxIndSets = np.empty((max_v,max_v),dtype=object)

# Load arrays
alphas = np.loadtxt("JohnsonAlphas.txt",dtype=dtype)
max_v = np.shape(alphas)[0]
maxIndSets = np.load("maxJohnsonIndSets.npy")

# Compute independent sets
for v in range(max_v):
    for k in range(v//2+1):

        # Get Johnson adjacency array
        G = genJohnsonGraph(v,k,k-1)
        A = getAdjArray(G)
        B = adjArray2List(A)

        # Find an independent set
        maxIndSetIndicator = fastFindIndSet(B,niters,ntrials,start=-2,stop=2.5)
        maxIndSet = np.array([[ord(c) for c in G.nodes()[i]] for i in np.where(maxIndSetIndicator)[0]],dtype=dtype)
        alpha = np.sum(maxIndSetIndicator)

        # If best so far...
        if isIndependent(maxIndSetIndicator,A):
            if alpha>alphas[v,k]:
                maxIndSets[v,k] = maxIndSet
                alphas[v,k] = alpha
                alphas[v,v-k] = alpha
                np.savetxt("JohnsonAlphas.txt",alphas,fmt='%5d')
                np.save("maxJohnsonIndSets.npy",maxIndSets)
        else:
            print("Not independent")    
