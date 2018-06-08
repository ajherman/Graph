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

dtype = "int32"
niters = 100 # Number of iterations

def sigma(X,T):
    return 1./(1+np.exp(-X/T))
    
def isIndependent(a,A):
    return np.dot(a,np.dot(A,a.T))==0
    
##################
# Adjacency matrix
##################

# Jvki
Jv = 'abcdefghij' # MUST BE IN ALPHABETICAL ORDER STARTING WITH A!!!
Jk = 4
Ji = 3
combos=list(it.combinations(Jv, Jk))
edges=[]
combos=[''.join(t[0] for t in x) for x in combos]
for i,x in enumerate(combos):
    for y in combos[i:]:
        if len(set(x) & set(y))==Ji:
            edges.append((x,y))
            edges.append((y,x))
G = nx.empty_graph(0, create_using=nx.DiGraph())
G.add_edges_from(e for e in edges)
A=np.array(nx.to_numpy_matrix(G)).astype(dtype)

## Random
#N = 20 # Number of vertices
#q = 0.3 # Edge probability
#A = np.random.random((N,N))
#A = 0.5*(A+A.T)
#A = 1*(A<q)
#A -= np.diag(np.diag(A))

## Complete graph
#A = np.ones((N,N))-np.eye(N)

######################
# Find independent set
######################
tic = tm.time()
N = np.shape(A)[0]
best_set = np.zeros((N,),dtype=dtype)
beta = 0
betas = []
for m in range(100): # Do best of 50 attempts
    a = np.zeros((N,),dtype=dtype) # Initial active nodes
    z = np.zeros((N,),dtype=dtype) 
    for k in np.linspace(-2,2,niters): # Run network while annealing temperature
        T = np.exp(-k)
        idx = [j for j in range(N)]
        random.shuffle(idx)
        for i in idx:
            new = np.random.random() < sigma(-2*z[i]+1,T)
            old = a[i]
            if old != new:     
                if new<old:
                    z -= A[i]
                    a[i] = new
                else:
                    z += A[i]
                    a[i] = new
    betas.append(np.sum(a))
    if np.sum(a)>beta:
        best_set = a
        beta = np.sum(a) 
    
toc = tm.time()
print("Run time: "+str(toc-tic)+"s")
print("Independent set computed by stochastic algorithm")
print(best_set)
print("Is it actually independent?")
if isIndependent(a,A):
    print("YES")
else:
    print("NO")
print("Independence number according to the stochastic algorithm: " + str(beta))


#prints the independent set
IS = np.array([[ord(c)-97 for c in G.nodes()[i]] for i in np.where(best_set)[0]],dtype=dtype)
'''
There's got to be a better way to do this part
'''
IS_bin = np.zeros((beta,len(Jv)),dtype=dtype)
for k in range(beta):
    IS_bin[k][IS[k]]=1
#print(IS,"\n") # Print indices
#print(IS_bin) # Print binary

# Histogram of independent set sizes (requires seaborn)
import seaborn as sns
sns.distplot(betas)
plt.show()

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

    

    