import numpy as np
from GraphFun import *

IS = np.load("IndependentSets/JvkiIndependentSets.npy")
for v in range(1,64):
    for k in range(1,v//2+1):
        for i in range(k):
            if not IS[v,k,i] is None:
                print(v,k,i)
                X = IS[v,k,i]
                n = len(X)
                adj = np.zeros((n,n),dtype='bool')
                for j1 in range(n):
                    for j2 in range(j1+1):
                        if np.dot(X[j1],X[j2])>=i:
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
                    print("Partition: ", sizes)

                    unions = np.zeros((len(sizes),v),dtype='int')
                    for s in range(len(sizes)):
                        for w in components[s]:
                            unions[s]+=X[w]
                    unions[unions!=0]=1
                    union_sizes = [np.sum(x) for x in unions]
                    print(union_sizes,'\n')

                    good = []
                    for s in range(len(sizes)):
                        for t in range(s):
                            is_less = np.dot(unions[s],unions[t])<i
                            if not is_less:
                                good.append(np.dot(unions[s],unions[t]))
                    if good!=[]:
                        print("Counter example: ")
                        print(good)
                        print('\n\n')
                                
                else:
                    print(X)
                    print(adj.astype('int'))
                    assert(0)
