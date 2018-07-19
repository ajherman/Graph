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
max_v = 25

# If starting from scratch (overwrites arrays if uncommented!)
omegas = -np.ones((max_v,max_v),dtype=dtype)
maxCliques = np.empty((max_v,max_v),dtype=object)

## Load arrays
#omegas = np.loadtxt("JohnsonOmegas.txt",dtype=dtype)
#max_v = np.shape(omegas)[0]
#maxCliques = np.load("maxJohnsonCliques.npy")

# Compute independent sets
#for v in range(max_v):
#    for k in range(v//2+1):
for v in range(20,23):
    for k in range(2,5):

        # Get Johnson adjacency array
        G = nx.complement(genJohnsonGraph(v,k,1))
        A = getAdjArray(G)
        B = adjArray2List(A)

        # Find an independent set
        maxCliqueIndicator = fastFindIndSet(B,niters,ntrials,start=-2,stop=2.5)
        maxClique = np.array([[ord(c) for c in G.nodes()[i]] for i in np.where(maxCliqueIndicator)[0]],dtype=dtype)
        omega = np.sum(maxCliqueIndicator)

        # If best so far...
        if isIndependent(maxCliqueIndicator,A):
            if omega>omegas[v,k]:
                maxCliques[v,k] = maxClique
                omegas[v,k] = omega
#                omegas[v,v-k] = omega
                np.savetxt("JohnsonOmegas.txt",omegas,fmt='%5d')
                np.save("maxJohnsonCliques.npy",maxCliques)
        else:
            print("Not independent")    
