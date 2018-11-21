# Make array of Johnson graph independence numbers

import numpy as np
import scipy as sc
import time as tm
from GraphFun import *

## If starting from scratch (overwrites arrays if uncommented!)
#alphas = -np.ones((64,64,64),dtype=dtype)
#maxIndSets = np.empty((64,64,64),dtype=object)

dtype = "int16"
niters = 5 # Number of iterations
ntrials = 1 # Number of times to repeat algorithm
max_v = 30

#idx_set = sorted([(v,k,i) for v in range(1,max_v) for k in range(1,v//2+1) for i in range(k)],key=lambda x: sc.special.binom(x[0],x[1])*sc.special.binom(x[1],x[2])*sc.special.binom(x[0]-x[1],x[1]-x[0])) # All Jvki graph in order of size
#idx_set = [(v,k,k-1) for v in range(13,23) for k in range(1,v//2+1)] # Johnson
#idx_set = [(max_v,k,i) for k in range(1,max_v) for i in range(k)] # Fixed v
#idx_set = [(v,k,1) for v in range(2,max_v) for k in range(2,v)] # Fixed i
#idx_set = [(v,4,2) for v in range(4,max_v)]
#idx_set = [(v,5,1) for v in range(5,max_v)]
#idx_set = [(v,8,1) for v in range(12,max_v)]
#idx_set = [(v,7,1) for v in range(16,max_v)]
idx_set = [(20,5,2)]

# Load arrays
alphas = np.load("IndependentSets/JvkiAlphas.npy")
maxIndSets = np.load("IndependentSets/JvkiIndependentSets.npy")

# Compute independent sets
for v,k,i in idx_set:

        # Get Johnson adjacency array
        V,B = genJohnsonAdjList(v,k,i)
        
        # Find an independent set
        maxIndSet = fastFindIndSetAlt(B,niters,ntrials,start=-2,stop=2.5)
        alpha = np.sum(maxIndSet,dtype=dtype)

        # If best so far...
        if alpha>alphas[v,k,i]:
            maxIndSets[v,k,i] = V[maxIndSet.astype(bool)]
            alphas[v,k,i] = alpha
            alphas[v,v-k,v-2*k+i] = alpha # Complement automorphism
            np.save("IndependentSets/JvkiAlphas.npy",alphas)
            np.save("IndependentSets/JvkiIndependentSets.npy",maxIndSets)

