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
import collections
from GraphFun import *

dtype = "int32"
niters = 10000 # Number of iterations


##################
# Adjacency matrix
##################
# Generalize Johnson graph
v,k,i = 8,4,1#7,3,1#13,5,4 #15,3,0 #17,8,0 #13,5,4 # 19,9,8
# V,A = genJohnsonAdjList(v,k,i)

V,A = genGenJohnsonAdjList(v,[k],[1]) #for j in range(0,i)]+[j for j in range(i+1,k)])
# print(A)
## Random
#N = 1000 # Number of vertices
#q = 0.99999 # Edge probability
#A = np.random.random((N,N))
#A = 0.5*(A+A.T)
#A = 1*(A<q)
#A -= np.diag(np.diag(A))
#
## Complete graph
#A = np.ones((N,N))-np.eye(N)




# most=collections.Counter([1,2,2,2,3,3]).most_common()
# print(most)
# [(2, 3), (3, 2), (1, 1)]
n=len(V)
near=0
minChi=np.inf
Cbest=np.array([0 for x in range(n)])
for ii in range(niters):
    C=np.array([0 for x in range(n)])
    order=np.random.permutation(n)
    for x in order:
        neigh=C[A[x]]
        mex=min(set(range(n))-set(neigh))
        C[x]=mex
        #for allowing near colorings
        least=collections.Counter(neigh).most_common()[::-1]
        for c in least:
            if c[0]<C[x] and c[1]<=near:
                C[x]=c[0]
    chitemp=np.size(np.unique(C))
    if chitemp<minChi:
        minChi=chitemp
        Cbest=C
print("near clique degree:", near)
print(minChi,Cbest)
print("alpha<=",minChi*(near+1))
# print(V)
# print(A)
