# Make array of Jvki graph independence numbers
# Can't take advantage of J(v,k,i)-J(v,v-k,v-2k+i) symmetry as easily, 
# so I wonder if it is instead worth it to save a 3D-array with values, and
#load and update that appropriately, and just make a function in Graph fun
#  that prints slices that we want to look at in GraphFun

import numpy as np
import random
import networkx as nx
import networkx.algorithms.approximation as nxaa
import matplotlib.pyplot as plt
import networkx.generators.directed
import itertools as it
import time as tm
#from GraphFun import *

dtype = "int32"
niters = 100 # Number of iterations
ntrials = 64 # Number of times to repeat algorithm
max_v = 14
ii=1#intersection number

## If starting from scratch (overwrites arrays if uncommented!)
#alphas = -np.ones((max_v,max_v),dtype=dtype)
# comment out this and all lines with maxIndSets if you only want the table.
#maxIndSets = np.empty((max_v,max_v),dtype=object)

# Load arrays
alphas = np.loadtxt("IndependentSets/Jvk"+str(ii)+"AlphasTentative.txt",dtype=dtype)
#max_v = np.shape(alphas)[0]
maxIndSets = np.load("IndependentSets/maxJvk"+str(ii)+"IndSetsTentative.npy")

# Compute independent sets
for v in range(max_v):
    print(v)
    for k in range(1,v+1):

        # Get Johnson adjacency array
        V,B = genJohnsonAdjList(v,k,ii)
        
        # Find an independent set
        maxIndSetIndicator = fastFindIndSet(B,niters,ntrials,start=-2,stop=2)
        maxIndSet = np.array([V[i] for i in np.where(maxIndSetIndicator)[0]],dtype=dtype)
        alpha = np.sum(maxIndSetIndicator)

        # If best so far...
        if alpha>alphas[v,k]:
            maxIndSets[v,k] = maxIndSet
            alphas[v,k] = alpha
            np.savetxt("IndependentSets/Jvk"+str(ii)+"AlphasTentative.txt",alphas,fmt='%5d',newline='\r\n')
            np.save("IndependentSets/maxJvk"+str(ii)+"IndSetsTentative.npy",maxIndSets)
