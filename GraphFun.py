import numpy as np
import random
import networkx as nx
import networkx.algorithms.approximation as nxaa
import matplotlib.pyplot as plt
import networkx.generators.directed
import itertools as it

dtype = "int32"

def sigma(X,T):
    return 1./(1+np.exp(-X/T))
    
def isIndependent(a,A):
    return np.dot(a,np.dot(A,a.T))==0
   
def genJohnsonGraph(v,k,i):
    vset = ''.join([chr(c) for c in range(v)]) # MUST BE IN ALPHABETICAL ORDER STARTING WITH A!!!
    combos=list(it.combinations(vset, k))
    edges=[]
    combos=[''.join(t[0] for t in x) for x in combos]
    for idx,x in enumerate(combos):
        for y in combos[idx:]:
            if len(set(x) & set(y))==i:
                edges.append((x,y))
                edges.append((y,x))
    G = nx.empty_graph(0, create_using=nx.DiGraph())
    G.add_edges_from(e for e in edges)
    return G

def getAdjArray(G):
    return np.array(nx.to_numpy_matrix(G)).astype(dtype)

def findIndSet(A,niters,start=-2,stop=2):
    N = np.shape(A)[0]
    a = np.zeros((N,),dtype=dtype) # Initial active nodes
    z = np.zeros((N,),dtype=dtype) # This is kept equal to A*a 
    for k in np.linspace(start,stop,niters): # Run network while annealing temperature
        T = np.exp(-k)
        idx = np.random.permutation(N)
        no_change = True # Does a change at all in this iteration?  This is not currently being used.
        for i in idx:
            new = np.random.random() < sigma(-2*z[i]+1,T)
            old = a[i]
            if old != new:     
                if new<old:
                    z -= A[i]
                else:
                    z += A[i]
                a[i] = new
                no_change = False
    return a

def fastFindIndSet(A,niters,ntrials,start=-2,stop=2): # I think this can still be vectorized better
    N = np.shape(A)[0]
    a = np.zeros((N,ntrials),dtype=dtype) # Initial active nodes
    z = np.zeros((N,ntrials),dtype=dtype) # This is kept equal to A*a 
    for k in np.linspace(start,stop,niters): # Run network while annealing temperature
        T = np.exp(-k)
        idx = np.random.permutation(N)
        for i in idx:
            new = np.random.random((ntrials,)) < sigma(-2*z[i]+1,T)
            old = a[i]
            sgn = np.sign(new-old)
            z = z + np.outer(A[i],sgn)
            a[i] = new
    aa = a[:,np.where(np.einsum('it,ij,jt->t',a,A,a)==0)[0]] # Consider only the sets that are actually independent
    idx = np.argmax(np.sum(aa,axis=0))
    return aa[:,idx]

def getIndependenceNumber(A,niters,ntrials,start=-2,stop=2):
    best = fastFindIndSet(A,niters,ntrials,start,stop)
    return np.sum(best)

def getFracChromNumber(A,niters,ntrials,start=-2,stop=2): # For vertex transitive graphs only!!!!!!!!!!!!
    return np.shape(A)[0]/getIndependenceNumber(A,niters,ntrials,start=start,stop=stop)
