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
niters = 100 #100000# Number of iterations
ntrials = 20 # Number of times to repeat algorithm

#niters = 30000 # gave 119
#ntrials = 25

##################
# Adjacency matrix
##################
# Generalize Johnson graph
v,k,i = 16,4,2 #15,8,3 #13,5,4 #15,3,0 #17,8,0 #13,5,4 # 19,9,8
V,A = genJohnsonAdjList(v,k,i)
#V,A=genKneserAdjList(v,k,i)

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
print(toc-tic)
print(beta)
s=V[best_set.astype(bool)]
count1=0
count3=0
for i in range(20):
    for j in range(i):
        r=np.dot(s[:,i],s[:,j])
        if r==1:
#            print(str(i)+" - " + str(j)) 
#            idx=np.where(s[:,i]*s[:,j]==1)[0][0]
#            print(np.where(s[idx]==1)[0])
            count1+=1
        elif r==3:
            count3+=1
        else:
            print(r)
            assert(0)
print(count1)
print(count3)
print('')
count0=0
count1=0
count3=0
for i in range(85):
    for j in range(i):
        r=np.dot(s[i],s[j])
        if r==0:
            count0+=1
        elif r==1:
            count1+=1
        elif r==3:
            count3+=1
        else:
            print(r)
            assert(0)
print(count0)
print(count1)
print(count3)
print('')

print(np.sum(s,axis=0))

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
