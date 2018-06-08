# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 15:14:19 2018

@author: TEA
"""

'''
Ari did some improvements to make this faster...
'''

# Finding independent sets with Boltzmann machines

import numpy as np
import random
import networkx as nx
import networkx.algorithms.approximation as nxaa
import matplotlib.pyplot as plt
import networkx.generators.directed
import itertools as it

def sigma(X,T):
    return 1./(1+np.exp(-X/T))
    
def isIndependent(a,A):
    return np.dot(a,np.dot(A,a.T))==0
    
##################
# Adjacency matrix
##################

# Jvki
Jv='123456789'
Jk=4
Ji=0
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
A=np.array(nx.to_numpy_matrix(G)).astype("int32")

## Random
#N = 20 # Number of vertices
#q = 0.3 # Edge probability
#A = np.random.random((N,N))
#A = 0.5*(A+A.T)
#A = 1*(A<q)
#A -= np.diag(np.diag(A))

## Peterson graph
#A = np.array([[0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
#[1, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
#[0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
#[0, 0, 1, 0, 1, 0, 0, 0, 1, 0], 
#[1, 0, 0, 1, 0, 0, 0, 0, 0, 1], 
#[1, 0, 0, 0, 0, 0, 0, 1, 1, 0], 
#[0, 1, 0, 0, 0, 0, 0, 0, 1, 1], 
#[0, 0, 1, 0, 0, 1, 0, 0, 0, 1], 
#[0, 0, 0, 1, 0, 1, 1, 0, 0, 0], 
#[0, 0, 0, 0, 1, 0, 1, 1, 0, 0]])

## Complete graph
#A = np.ones((N,N))-np.eye(N)

######################
# Find independent set
######################

N = np.shape(A)[0]
best_set = np.zeros((N,))
beta = 0
itera=30
decreased=[0 for x in range(itera)]
fixed=[() for x in range(itera)]
for m in range(itera): # Do best of 50 attempts
    a = 1*(np.random.random((N,))<0.5) # Initial active nodes
    z = np.dot(a,A)
    fixer=0# strings of fixed size independent sets
#    decrease=0 #strings of decreseing ind sets, seems to not happen k>1
    fixer=0
    for k in range(-30,200): # Run network while annealing temperature
        aprev=a
        T = np.exp(-k)
        idx = [j for j in range(N)]
        random.shuffle(idx)
        for i in idx:
            p = sigma(-2*z[i]+1,T)
            new = np.random.random() < p
            old = a[i]
            if new<old:
                z -= A[i,:]
                a[i] = new
            elif new>old:
                z += A[i,:]
                a[i] = new
            else:
                pass
#        if k>1 and np.sum(a)<np.sum(aprev):
#            a=aprev
#            decreased[m]=k
#            break
        if k>0 and np.sum(a)==np.sum(aprev):
            fixer+=1
            if fixer>5:
                fixed[m]=(k, fixer)
                print("Final iteration: "+str(k))
                break
        elif k>1:
            fixer=0
    if np.sum(a)>beta:
        best_set = a
        beta = np.sum(a) 

print(decreased)
print(fixed,'\n')
#print("Adjacency matrix")
#print(A)
print("Independent set computed by stochastic algorithm")
print(best_set)
print("Is it actually independent?")
if isIndependent(a,A):
    print("YES")
else:
    print("NO")
print("Independence number according to the stochastic algorithm: " + str(beta))

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

    

    