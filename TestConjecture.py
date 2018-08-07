import numpy as np
from GraphFun import *
import time
import scipy.special

IS = np.load("IndependentSets/JvkiIndependentSets.npy")
for v in range(1,64):
    for k in range(1,v//2+1):
        for i in range(k):
            v,k,i = 15,6,3
            print(v,k,i)
            if not IS[v,k,i] is None:
                X = IS[v,k,i]
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
                    print(len(X))
                    for c in components:
                        x = X[c] 
                        if len(x)>1:
                            s=np.sum(x,axis=0)
                            avg = (np.min(s)+np.max(s))/2.0
                            tt=list(np.where(s>avg)[0])
                            rr=list(np.where(s==0)[0])

                            M = len(tt)
                            N = v-M-len(rr)
                            r = np.sum(x[:,tt],axis=1)
                            m = np.min(r)
                            
                            print('\n\n')
                            print(len(x))
                            print(s)
                            print(len(x))
                            print(M,':',m)
                            print(N)
                            print('')

                            if 2*m-M-i<=0   and not (v==2*k and i==0):
                                print(v,k,i)
                                assert(0)

#                            print('') 
#                            for tt in types:
#                                r=np.sum(x[:,tt],axis=1)
#                                m=np.min(r)
#                                print(len(tt),':',set(r))
#                                M=np.max(r)
#                                total += max(0,2*m-len(tt))
#                            if total<=i and not (v==2*k and i==0):
#                                print(v,k,i)
#                                assert(0)

#                    print("Partition: ", sizes)
#                    print("Partition: ", components)
                    assert(0) 
                else:
                    print(X)
                    print(adj.astype('int'))
                    assert(0)
