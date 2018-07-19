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
niters = 10000 #100000# Number of iterations
ntrials = 20 # Number of times to repeat algorithm

#niters = 30000 # gave 119
#ntrials = 25

##################
# Adjacency matrix
##################
# Generalize Johnson graph
v,k,i = 9,4,0 #13,5,4 #17,8,0 #13,5,4 # 19,9,8
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
best_set,oth_ind = fastFindIndSetExp(A,niters,ntrials,anneal=3,otherind=True)
beta = np.sum(best_set)
toc = tm.time()
print("Run time for fast version: "+str(toc-tic)+"s")
print("Independence number according to the stochastic algorithm: " + str(beta))
#print("Best set: ")
#print(best_set)
IS = np.array([np.where(V[i])[0] for i in np.where(best_set)[0]],dtype=dtype)
print("Independent set computed by stochastic algorithm")
print(IS,"\n") # Print indices

print("Sizes of independent sets found :")
print(oth_ind)     

bestbinary=np.array([V[i] for i in np.where(best_set)[0]])
print("number of pairwise intersections sizes *2? :")
print(np.bincount(np.reshape(np.dot(bestbinary,bestbinary.T),len(bestbinary)**2).astype(int)))
'''
There's got to be a better way to do this part
'''
# if you want to save the independent set in the binary form
#np.savetxt("binary.txt",bestbinary,fmt='%i',newline='\r\n')

# if you want to save the pairwise intersection matrix
#np.savetxt("best121j13-5-4.txt",np.dot(bestbinary,bestbinary.T),fmt='%i',newline='\r\n')

#IS_bin = np.zeros((beta,v),dtype=dtype)
#for k in range(beta):
#    IS_bin[k][IS[k]]=1
#print(IS_bin) # Print binary

## Histogram of independent set sizes (requires seaborn)
#import seaborn as sns
#sns.distplot(betas)
#plt.show()


