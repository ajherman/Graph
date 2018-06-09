# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 15:14:19 2018

@author: TEA
"""

# Finding independent sets with Boltzmann machines

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
ntrials = 50

##################
# Adjacency matrix
##################

# Jvki
v,k,i = 7,3,1
G = genJohnsonGraph(v,k,i)
A = getAdjArray(G)

## Random
#N = 20 # Number of vertices
#q = 0.3 # Edge probability
#A = np.random.random((N,N))
#A = 0.5*(A+A.T)
#A = 1*(A<q)
#A -= np.diag(np.diag(A))

## Complete graph
#A = np.ones((N,N))-np.eye(N)

# Get complement adjacenty array
Ac = 1-A # Complement graph
np.fill_diagonal(Ac,0)

#############
# Find Clique
#############
tic = tm.time()
N = np.shape(A)[0]
best_set = np.zeros((N,),dtype=dtype)
beta = 0
betas = []
for m in range(ntrials): # Do best of 50 attempts
    a = findIndSet(Ac,niters,start=-2,stop=2)
    if isIndependent(a,Ac):
        betas.append(np.sum(a))
        if np.sum(a)>beta:
            best_set = a
            beta = np.sum(a)  
toc = tm.time()

# Print independent set
print("Run time: "+str(toc-tic)+"s")
print("Clique computed by stochastic algorithm")
print(best_set)
print("Clique number according to the stochastic algorithm: " + str(beta))


IS = np.array([[ord(c) for c in G.nodes()[i]] for i in np.where(best_set)[0]],dtype=dtype)
print("Independent set computed by stochastic algorithm")
print(IS,"\n") # Print indices

