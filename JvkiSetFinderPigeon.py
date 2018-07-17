# Make array of Jvki graph independence numbers
# THis takes advantage of the fact that if  k>v//2+i//2+1 then
# J(v,k,i) will have no indpendent sets, I think?
#however it wont produce the actual indpendent sets, to save time,
# it would just be all vertices in that case, so we could have it do that later


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
import scipy.special as scis
import time as tm
#from GraphFun import *

dtype = "int32"
niters = 128 # Number of iterations
ntrials = 64 # Number of times to repeat algorithm
max_v = 13
ii=1#intersection number

## If starting from scratch (overwrites arrays if uncommented!)
alphas = -np.ones((max_v,max_v),dtype=dtype)
## comment out this and all lines with maxIndSets if you only want the table.
maxIndSets = np.empty((max_v,max_v),dtype=object)

# Load arrays
#alphas = np.loadtxt("IndependentSets/Jvk"+str(ii)+"AlphasPigeon.txt",dtype=dtype)
#max_v = np.shape(alphas)[0]
#maxIndSets = np.load("IndependentSets/maxJvk"+str(ii)+"IndSetsPigeon.npy")

# Compute independent sets
for v in range(ii,max_v):
    print(v)
    for k in range(ii,v//2+ii//2+2):

        # Get Johnson adjacency array
        V,B = genJohnsonAdjList(v,k,ii)
        
        # Find an independent set
        maxIndSetIndicator = fastFindIndSetExp(B,niters,ntrials,anneal=3,start=-2,stop=2)
#        maxIndSet = np.array([V[i] for i in np.where(maxIndSetIndicator)[0]],dtype=dtype)
        alpha = np.sum(maxIndSetIndicator)

        # If best so far...
        if alpha>alphas[v,k]:
#            maxIndSets[v,k] = maxIndSet
            alphas[v,k] = alpha
            np.savetxt("IndependentSets/Jvk"+str(ii)+"AlphasPigeon.txt",alphas,fmt='%5d',newline='\r\n')
#            np.save("IndependentSets/maxJvk"+str(ii)+"IndSetsTentative.npy",maxIndSets)
    for k in range(v//2+ii//2+2,v+1):
        alphas[v,k] = scis.binom(v, k)
        np.savetxt("IndependentSets/Jvk"+str(ii)+"AlphasPigeon.txt",alphas,fmt='%5d',newline='\r\n')