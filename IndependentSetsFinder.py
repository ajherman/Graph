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
niters = 200 # Number of iterations
ntrials = 60 # Number of times to repeat algorithm

##################
# Adjacency matrix
##################
# Generalize Johnson graph
v,k,i = 23,4,3

G = genJohnsonGraph(v,k,i)
A = getAdjArray(G)

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
B = adjArray2List(A)
best_set = fastFindIndSet(B,niters,ntrials)
beta = np.sum(best_set)
toc = tm.time()
print("Run time for fast version: "+str(toc-tic)+"s")
print("Independence number according to the stochastic algorithm: " + str(beta))
IS = np.array([[ord(c) for c in list(G)[i]] for i in np.where(best_set)[0]],dtype=dtype)
print("Independent set computed by stochastic algorithm")
print(IS,"\n") # Print indices
print(best_set)

## Fast version
#tic = tm.time()
#best_set2 = fastFindIndSet(A,niters,ntrials,adjlist=False)
#beta2 = np.sum(best_set2)
#toc = tm.time()
#print("Run time for fast version: "+str(toc-tic)+"s")
#print("Independence number according to the stochastic algorithm: " + str(beta2))
#IS = np.array([[ord(c) for c in list(G)[i]] for i in np.where(best_set2)[0]],dtype=dtype)
#print("Independent set computed by stochastic algorithm")
#print(IS,"\n") # Print indices
#print(best_set2)
#
## Slow version
#tic = tm.time()
#N = np.shape(A)[0]
#best_set = np.zeros((N,),dtype=dtype)
#beta = 0
#betas = []
#for m in range(ntrials): # Do best of 50 attempts
#    a = findIndSet(A,niters)
#    if isIndependent(a,A):
#        betas.append(np.sum(a))
#        if np.sum(a)>beta:
#            best_set = a
#            beta = np.sum(a) 
#    else:
#        print("Not independent")
#toc = tm.time()
#print("Run time: "+str(toc-tic)+"s")
#print("Independence number according to the stochastic algorithm: " + str(beta))
#IS = np.array([[ord(c) for c in list(G)[i]] for i in np.where(best_set)[0]],dtype=dtype)
#
#print("Independent set computed by stochastic algorithm")
#print(IS,"\n") # Print indices


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

## Optional other stuff
#G = nx.from_numpy_matrix(A)
#gamma = len(nxaa.maximum_independent_set(G))
#print("Independent number according to networkx algorithm: " + str(gamma))
#
## Get exact independence number (comment this section out if N is large, because it will take forever)
#alpha = 0
#for t in it.product([0, 1], repeat=N):
#    a = np.array(t).reshape(1,N)
#    if isIndependent(a,A):
#        alpha = max(alpha,np.sum(a))
#print("Exact indpendence number is: " + str(alpha))

    

    
