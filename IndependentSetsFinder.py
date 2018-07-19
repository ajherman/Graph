# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 15:14:19 2018

@author: TEA
"""

# Finding independent sets with Boltzmann machines

import numpy as np
import random
#import networkx as nx
#import networkx.algorithms.approximation as nxaa
#import matplotlib.pyplot as plt
#import networkx.generators.directed
import itertools as it
import time as tm
from GraphFun import *

dtype = "int32"
niters = 100 # Number of iterations
ntrials = 100 # Number of times to repeat algorithm

##################
# Adjacency matrix
##################
# Generalize Johnson graph
v,k,i = 13,5,4 #17,8,0 #13,5,4 # 19,9,8
V,A = genJohnsonAdjList(v,k,i)

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

######################
# Find independent set
######################
# Super fast version
tic = tm.time()
best_set = fastFindIndSetAlt(A,niters,ntrials,-2,2)
beta = np.sum(best_set)
toc = tm.time()
print("Run time for fast version: "+str(toc-tic)+"s")
print("Independence number according to the stochastic algorithm: " + str(beta))
#print("Best set: ")
#print(best_set)
IS = np.array([np.where(V[i])[0] for i in np.where(best_set)[0]],dtype=dtype)
print("Independent set computed by stochastic algorithm")
print(IS,"\n") # Print indices

'''
There's got to be a better way to do this part
'''
#IS_bin = np.zeros((beta,v),dtype=dtype)
#for k in range(beta):
#    IS_bin[k][IS[k]]=1
#print(IS_bin) # Print binary

## Histogram of independent set sizes (requires seaborn)
#import seaborn as sns
#sns.distplot(betas)
#plt.show()


