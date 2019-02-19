'''
The getInfo() function is embarrassingly inefficient, but it does what we want.  Basically, if you give it a v,k,i, it will return an array whose first row are the sizes of blocks of [v] (a partition) and whose other rows give you options for how many things to take from each block.  For all of the cases I've tested (except J(2k,k,0)) this is always a full description...meaning that every possible combination is actually used and together they account for the entire independent set.  So, that's cool!  Basically, these give a more compact way to discuss one of these independent sets.  However, some of them are still pretty big (particularly Johnson graphs) and I think there may be further tricks to make the descriptions even more compact.  
'''

import numpy as np
from GraphFun import *
import time
import scipy.special



def getInfo(v,k,i):
    X=np.loadtxt("j11-5-1bin.txt")
    n = len(X)
    adj = np.zeros((n,n),dtype='bool')
    for j1 in range(n):
        for j2 in range(j1+1):
            r = np.dot(X[j1],X[j2])
            if r>=i:
                adj[j1,j2]=1
                adj[j2,j1]=1
    val = ~np.any(np.dot(adj,adj)[np.where(~adj)])
    if val:
        components = []
        reached = np.array([False]*n)
        while not np.all(reached):
            stack = []
            components.append([])
            for w in range(n):
                if not reached[w]:
                    stack.append(w)
                    reached[w]=True
                    break 
            while stack!=[]:
                w1 = stack.pop()
                components[-1].append(w1)
                neighbors = np.where(adj[w1])[0]
                for w2 in neighbors:
                    if not reached[w2]:
                        stack.append(w2)
                        reached[w2]=True
        sizes = [len(c) for c in components]
        types = []
        for c in components:
            x = X[c] 
            s=np.sum(x,axis=0)
            types.append(s)
        types = np.stack(types,axis=0)
        s = set([])
        for tt in types.T:
            s.add(tuple(tt))
        s = list(s)
        blocks = [[] for ii in range(len(s))]
        for w in range(v):
            tt=tuple(types.T[w])
            for r in range(len(s)):
                if s[r]==tt:
                    blocks[r].append(w)
        
        classes = []
        for b in blocks:
            classes.append(np.sum(X[:,b],axis=1))
        classes=np.stack(classes).T
        classes = set([tuple(c) for c in classes])
        blocks = [len(b) for b in blocks]
       
        some = 0
        for c in classes:
            prod = 1
            for idx in range(len(blocks)):
                prod*=scipy.special.binom(blocks[idx],c[idx])
            some += prod
        if some!=len(X):
            print("Independent set was not 'full'")
            assert(0)

        complete = [blocks]+[list(c) for c in classes]

        return np.array(complete)

v,k,i=11,5,1
blocks = getInfo(v,k,i)
print(v,k,i)
print(blocks)
print("")
