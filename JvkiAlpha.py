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

## If starting from scratch (overwrites arrays if uncommented!)
#alphas = -np.ones((64,64,64),dtype=dtype)
#maxIndSets = np.empty((64,64,64),dtype=object)

dtype = "int32"
niters = 100 # Number of iterations
ntrials = 100 # Number of times to repeat algorithm
max_v = 23
#idx_set = [(v,k,i) for v in range(1,max_v) for k in range(1,v//2+1) for i in range(k)] # All
#idx_set = [(v,k,k-1) for v in range(13,23) for k in range(1,v//2+1)] # Johnson
#idx_set = [(max_v,k,i) for k in range(1,max_v) for i in range(k)] # Fixed v
idx_set = [(v,k,1) for v in range(2,max_v) for k in range(2,v)] # Fixed i

# Load arrays
alphas = np.load("IndependentSets/JvkiAlphas.npy")
maxIndSets = np.load("IndependentSets/JvkiIndependentSets.npy")

# Compute independent sets
for v,k,i in idx_set:

        # Get Johnson adjacency array
        V,B = genJohnsonAdjList(v,k,i)
        
        # Find an independent set
        maxIndSet = fastFindIndSetAlt(B,niters,ntrials,start=-2,stop=2.5)
        alpha = np.sum(maxIndSet)

        # If best so far...
        if alpha>alphas[v,k,i]:
            maxIndSets[v,k,i] = maxIndSet
            alphas[v,k,i] = alpha
            alphas[v,v-k,v-2*k+i] = alpha # Complement automorphism
            np.save("IndependentSets/JvkiAlphas.npy",alphas)
            np.save("IndependentSets/JvkiIndependentSets.npy",maxIndSets)

        # Save txt file showing a specific slice of values
        display_alphas = alphas[sliceIdx('johnson',max_v)]
        np.savetxt("IndependentSets/AlphaArray.txt",display_alphas,fmt='%5d')

