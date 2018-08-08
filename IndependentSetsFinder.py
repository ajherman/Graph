#
#
## -*- coding: utf-8 -*-
#"""
#Created on Mon Feb 26 15:14:19 2018
#
#@author: TEA
#"""
#
## Finding independent sets with Boltzmann machines
#
#import numpy as np
#import random
##import networkx as nx
##import networkx.algorithms.approximation as nxaa
##import matplotlib.pyplot as plt
##import networkx.generators.directed
#import itertools as it
#import time as tm
#from GraphFun import *
#
#dtype = "int32"
#niters = 128 #100000# Number of iterations
#ntrials = 32 # Number of times to repeat algorithm
#
##niters = 30000 # gave 119
##ntrials = 25
#
###################
## Adjacency matrix
###################
## Generalize Johnson graph
#v,k,i = 15,6,4#8,4,1#16,8,1 #15,3,0 #17,8,0 #13,5,4 # 19,9,8
##nned to do Generlized kneser 18,6,2
## Generalize Johnson graph
##V,A = genJohnsonAdjList(v,k,i)
#
## Generalize kneser graph
#V,A=genKneserAdjList(v,k,i)
#
### Random
##N = 1000 # Number of vertices
##q = 0.99999 # Edge probability
##A = np.random.random((N,N))
##A = 0.5*(A+A.T)
##A = 1*(A<q)
##A -= np.diag(np.diag(A))
##
### Complete graph
##A = np.ones((N,N))-np.eye(N)
#
#=======
## -*- coding: utf-8 -*-
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
niters = 200 #100000# Number of iterations
ntrials = 50 # Number of times to repeat algorithm

#niters = 30000 # gave 119
#ntrials = 25

##################
# Adjacency matrix
##################
# Generalize Johnson graph
v,k,i = 15,(6,),(3,) #15,8,3 #13,5,4 #15,3,0 #17,8,0 #13,5,4 # 19,9,8
tic = time.time()
V,A = genGenJohnsonAdjList(v,k,i)
toc = time.time()
print("Time to build adjacency list: " + str(toc-tic))

## Random
#N = 1000 # Number of vertices
#q = 0.99999 # Edge probability
#A = np.random.random((N,N))
#A = 0.5*(A+A.T)
#A = 1*(A<q)
#A -= np.diag(np.diag(A))

######################
# Find independent set
######################
# Super fast version
tic = tm.time()
#best_set,oth_ind = fastFindIndSetExp(A,niters,ntrials,anneal=4,otherind=True)
best_set = fastFindIndSetAlt(A,niters,ntrials)
beta = np.sum(best_set)
toc = tm.time()
print("Time to compute independent set: "+str(toc-tic))
print("Independence number: " + str(beta))
s=V[best_set.astype(bool)]
print("Independent set: ")
print(s)

with open('output.txt', 'w') as f:
    def my_print(stri,stri2=''):
        stri=str(stri)
        stri2=str(stri2)
        f.write(stri+stri2 + '\n')
    
    my_print("Run time for fast version: "+str(toc-tic)+"s")
    my_print("Independence number according to the stochastic algorithm: " + str(beta))
    #print("Best set: ")
    #print(best_set)
    IS = np.array([np.where(V[i])[0] for i in np.where(best_set)[0]],dtype=dtype)
    my_print("Independent set computed by stochastic algorithm")
    my_print(IS,"\n") # Print indices
#    print(V[best_set.astype(bool)]) # Prints independent set in binary form
    my_print("Sizes of independent sets found :")
#    my_print(oth_ind)     
    
    bestbinary=np.array([V[i] for i in np.where(best_set)[0]])
    my_print("number of pairwise intersections sizes *2? :")
    my_print(np.bincount(np.reshape(np.dot(bestbinary,bestbinary.T),len(bestbinary)**2).astype(int)))
#>>>>>>> d50f11fc6cce0441d22598501be3ad8ab45a2a6b
#######################
## Find independent set
#######################
## Super fast version
#tic = tm.time()
#best_set,oth_ind = fastFindIndSetAlt(A,niters,ntrials,otherind=True)
#beta = np.sum(best_set)
#toc = tm.time()
#print("Run time for fast version: "+str(toc-tic)+"s")
#print("Independence number according to the stochastic algorithm: " + str(beta))
##print("Best set: ")
##print(best_set)
#IS = np.array([np.where(V[i])[0] for i in np.where(best_set)[0]],dtype=dtype)
#print("Independent set computed by stochastic algorithm")
#print(IS,"\n") # Print indices
#
#print("Sizes of independent sets found :")
#print(oth_ind)     
#
#bestbinary=np.array([V[i] for i in np.where(best_set)[0]])
#print("number of pairwise intersections sizes *2? :")
#print(np.bincount(np.reshape(np.dot(bestbinary,bestbinary.T),len(bestbinary)**2).astype(int)))
