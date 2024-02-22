# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 18:35:17 2018

@author: Ari
"""

# Finite field construction

import itertools as it
import numpy as np
from scipy.misc import comb
from matplotlib import pyplot as plt

Q = [17,23,29,31,37,41]
r = 8
s = 5

flower_sizes = []
level_sizes = []

# Map from r-set to cell indices    
def readable(w):
    return ''.join([chr(s+97) for s in w])

for q in Q:
    
    # Our finite field
    F = np.arange(q,dtype=np.int32)    
    
    # All r-subsets of F (i.e. vertices)
    V = list(it.combinations(F,r)) 

    # First r-s powers of F
    powF = np.array([F**d%q for d in range(1,r-s+1)])
     
    # Create empty arrays for cliques and clique sizes
#    clique_cover = np.zeros(shape,dtype=object)
    clique_sizes = np.zeros(shape = (q,)*(r-s),dtype=np.int32) # To store clique sizes
    
    # Get color classes 
    for w in V:
        idx = tuple(np.sum(powF[:,w],axis=1)%q) # Compute index of cell for w
    #    if clique_cover[idx]==0:
    #        clique_cover[idx] = []
    #    clique_cover[idx].append(readable(w))
        clique_sizes[idx]+=1
    
    # List of the various cell sizes that occur
    cell_sizes = list(set(clique_sizes.reshape(-1)))
    cell_sizes.sort()
 
    print("\n=======================================================")
    
    print("\n X = J("+str(q)+","+str(r)+","+str(s)+")\n")
    
    flower_sizes.append(comb(q,r-s-1))
    level_sizes.append(np.max(cell_sizes))
    if not 0 in cell_sizes:
        print("There are no empty cells")
        
    for cs in cell_sizes:
        n_cells = np.sum(clique_sizes==cs)
        print("Number of cells of size",cs,":",n_cells)
    
plt.plot(Q,flower_sizes,label="flowers")
plt.plot(Q,level_sizes,label="level sets")
plt.legend()
plt.show()
    
    
    
    
    
