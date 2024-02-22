# Make array of Johnson graph independence numbers

import numpy as np
import random
import networkx as nx
import networkx.algorithms.approximation as nxaa
import matplotlib.pyplot as plt
import networkx.generators.directed
import itertools as it
import time as tm
from chromatic.GraphFun import *

dtype = "int32"
niters = 100 # Number of iterations
ntrials = 64 # Number of times to repeat algorithm
max_v = 10

## If starting from scratch (overwrites arrays if uncommented!)
#alphas = -np.ones((max_v,max_v),dtype=dtype)
#maxIndSets = np.empty((max_v,max_v),dtype=object)

# Load arrays
alphas = np.loadtxt("IndependentSets/JAlphasTable.txt",dtype=dtype)
#max_v = np.shape(alphas)[0]
#maxIndSets = np.load("IndependentSets/maxJohnsonIndSetsTentative.npy")

# Compute independent sets
for v in range(max_v):
    print(v)
    for k in range(1,v//2+1):

        # Get Johnson adjacency array
        V,B = genJohnsonAdjList(v,k,k-1)
        
        # Find an independent set
        maxIndSetIndicator = fastFindIndSet(B,niters,ntrials,start=-2,stop=2.5)
        #maxIndSet = np.array([V[i] for i in np.where(maxIndSetIndicator)[0]],dtype=dtype)
        alpha = np.sum(maxIndSetIndicator)

        # If best so far...
        if alpha>alphas[v,k]:
            #maxIndSets[v,k] = maxIndSet
            alphas[v,k] = alpha
            alphas[v,v-k] = alpha
            np.savetxt("IndependentSets/JAlphasTable.txt",alphas,fmt='%5d',newline='\r\n')
#            np.save("IndependentSets/maxJohnsonIndSetsTentative.npy",maxIndSets)
