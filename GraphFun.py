import numpy as np
import random
#import networkx as nx
#import networkx.algorithms.approximation as nxaa
#import matplotlib.pyplot as plt
#import networkx.generators.directed
import itertools as it
import time

dtype = "int8"

def sigma(X,T): # Logistic function
    return 1./(1+np.exp(-X/T))
    
def isIndependent(a,A): # Tests if vertices specified by binary vector, a, are independent for adjacency array, A
    return np.dot(a,np.dot(A,a.T))==0

def genJohnsonAdjList(v,k,i,use_old_version=False): # Create J(v,k,i) graph
    if use_old_version:
        vset = ''.join([chr(c) for c in range(v)]) 
        combos=list(it.combinations(vset, k))
        V=[''.join(t[0] for t in x) for x in combos]
        adjList = [[] for v in V] 
        for idx1,x in enumerate(combos):
            for idx2,y in enumerate(combos):
                if len(set(x) & set(y))==i:
                    adjList[idx1].append(idx2)
    else:
        b = np.array([2**i for i in range(v)],dtype='uint64')
        combos1=list(it.combinations([ii for ii in range(k)], i))
        combos2=list(it.combinations([ii for ii in range(k,v)],k-i))
        allcombos=[x+y for x in combos1 for y in combos2]
        flip = np.zeros((len(allcombos),v),dtype=dtype)
        for idx,c in enumerate(allcombos):
            flip[idx,c]=1
        neighbors=np.zeros(np.shape(flip),dtype=dtype)
        vset = [ii for ii in range(v)]
        combos=list(it.combinations(vset, k))
        V = np.zeros((len(combos),v),dtype=dtype)
        for idx,c in enumerate(combos):
            V[idx,c]=1 
        hsh = {np.dot(x,b):ii for ii,x in enumerate(V)}
        adjList = [[] for v in V] 
        for x in V:
            idx = np.concatenate([np.where(x==1)[0],np.where(x==0)[0]])
            neighbors[:,idx] = flip
            key_x = np.dot(x,b)
            neighbor_keys = np.dot(neighbors,b)
            for key_y in neighbor_keys:
                adjList[hsh[key_x]].append(hsh[key_y]) 
    return V,adjList

def genJohnsonGraph(v,k,i): # Create J(v,k,i) graph
    vset = ''.join([chr(c) for c in range(v)]) 
    combos=list(it.combinations(vset, k))
    edges=[]
    combos=[''.join(t[0] for t in x) for x in combos]
    G = nx.empty_graph(0, create_using=nx.DiGraph())
    G.add_nodes_from(combos)
    for idx,x in enumerate(combos):
        for y in combos[idx:]:
            if len(set(x) & set(y))==i:
                G.add_edge(x,y)
                G.add_edge(y,x)
    return G

def genGKneserAdjList(v,k,i,use_old_version=True): # Create GK(v,k,i) graph only works for old version
    if use_old_version:
        vset = ''.join([chr(c) for c in range(v)]) 
        combos=list(it.combinations(vset, k))
        V=[''.join(t[0] for t in x) for x in combos]
        adjList = [[] for v in V] 
        for idx1,x in enumerate(combos):
            for idx2,y in enumerate(combos):
                if len(set(x) & set(y))<i:
                    adjList[idx1].append(idx2)
    else:
        b = np.array([2**i for i in range(v)],dtype='uint64')
        combos1=list(it.combinations([ii for ii in range(k)], i))
        combos2=list(it.combinations([ii for ii in range(k,v)],k-i))
        allcombos=[x+y for x in combos1 for y in combos2]
        flip = np.zeros((len(allcombos),v),dtype=dtype)
        for idx,c in enumerate(allcombos):
            flip[idx,c]=1
        neighbors=np.zeros(np.shape(flip),dtype=dtype)
        vset = [ii for ii in range(v)]
        combos=list(it.combinations(vset, k))
        V = np.zeros((len(combos),v),dtype=dtype)
        for idx,c in enumerate(combos):
            V[idx,c]=1 
        hsh = {np.dot(x,b):ii for ii,x in enumerate(V)}
        adjList = [[] for v in V] 
        for x in V:
            idx = np.concatenate([np.where(x==1)[0],np.where(x==0)[0]])
            neighbors[:,idx] = flip
            key_x = np.dot(x,b)
            neighbor_keys = np.dot(neighbors,b)
            for key_y in neighbor_keys:
                adjList[hsh[key_x]].append(hsh[key_y]) 
    return V,adjList

def adjArray2List(A): # Creates adjacency list from array
    N = np.shape(A)[0]
    B = []
    for j in range(N):
        B.append(np.where(A[j])[0])
    return B

def getAdjArray(G): # Create adjacency array from graph 
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

def fastFindIndSet(A,niters,ntrials,start=-2,stop=2,adjlist=True,otherind=False): 
    N = np.shape(A)[0]
    a = np.zeros((N,ntrials),dtype=dtype) # Initial active nodes
    z = np.zeros((N,ntrials),dtype=dtype) # This is kept equal to A*a 
    Ts = list(np.exp(-np.linspace(start,stop,niters-2)))
    Ts += [1e-10,1e-10]
    for itr,T in enumerate(Ts): # Run network while annealing temperature
#        print("Iteration: " + str(itr))
        idx = np.random.permutation(N)
        for i in idx:
            rando = np.random.random((ntrials,))
            s = sigma(-2*z[i]+1,T)
            new = rando < s
            delta = new-a[i]
            a[i] = new
            if adjlist:
                z[A[i]] += delta
            else:
                z += np.outer(A[i],delta)
    if otherind:
        return a[:,np.argmax(np.sum(a,axis=0))], np.sum(a,0)
    else:
        return a[:,np.argmax(np.sum(a,axis=0))]

def fastFindIndSetAlt(A,niters,ntrials,start=-2,stop=2): # Fastest version so far, I think 
    N = np.shape(A)[0]
    degree = np.shape(A)[1]
    a = np.zeros((N,ntrials),dtype=dtype) # Initial active nodes
    z = np.zeros((N,ntrials),dtype=dtype) # This is kept equal to A*a 
    Ts = list(np.exp(-np.linspace(start,stop,niters-2)))
    Ts += [1e-10,1e-10]
    for itr,T in enumerate(Ts): # Run network while annealing temperature
        precomputesigma = sigma(-2*np.arange(degree+1)+1,T)
        randos = np.random.random((N,ntrials))
        idx = np.random.permutation(N)
        for i in idx:
            new = randos[i] < precomputesigma[z[i]]
            delta = new-a[i]
            a[i] = new
            z[A[i]] += delta
    return a[:,np.argmax(np.sum(a,axis=0))]

def fastFindIndSetExp(A,niters,ntrials,start=-2,stop=2,adjlist=True,anneal=0,otherind=False): 
    N = np.shape(A)[0]
    a = np.zeros((N,ntrials),dtype=dtype) # Initial active nodes
    z = np.zeros((N,ntrials),dtype=dtype) # This is kept equal to A*a 
    Ts = list(np.exp(-np.linspace(start,stop,niters-2)))
    Ts += [1e-10,1e-10]
    Ts+=Ts[-20:-1]*anneal
    for itr,T in enumerate(Ts): # Run network while annealing temperature
#        print("Iteration: " + str(itr))
        idx = np.random.permutation(N)
        for i in idx:
            rando = np.random.random((ntrials,))
            s = sigma(-2*z[i]+1,T)
            new = rando < s
            delta = new-a[i]
            a[i] = new
            if adjlist:
                z[A[i]] += delta
            else:
                z += np.outer(A[i],delta)
    if otherind:
        return a[:,np.argmax(np.sum(a,axis=0))], np.sum(a,0)
    else:
        return a[:,np.argmax(np.sum(a,axis=0))]
    
def getIndependenceNumber(A,niters,ntrials,start=-2,stop=2):
    best = fastFindIndSet(A,niters,ntrials,start,stop)
    return np.sum(best)

#By Godsil page 142
def getFracChromNumber(A,niters,ntrials,start=-2,stop=2): # For vertex transitive graphs only!!!!!!!!!!!!
    return np.shape(A)[0]/getIndependenceNumber(A,niters,ntrials,start=start,stop=stop)

def drawGraph(G,layout='spectral',layout_array=None): # Taiyo's draw function
    if layout=='spectral':
        pos = nx.layout.spectral_layout(G)
    elif layout=='spring':
        pos = nx.layout.spring_layout(G,pos=nx.circular_layout(G),iterations=2)
    elif layout=='shell':
        pos = nx.layout.shell_layout(G,layout_array)
    node_sizes = [3 + 10 * i for i in range(len(G))]
    M = G.number_of_edges()
    edge_colors = [i for i in range(2, M + 2)]
    #nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='blue')
    edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='->',
                                   arrowsize=10, edge_color=edge_colors,
                                   edge_cmap=plt.cm.Blues, width=2)
    nx.draw_networkx_labels(G,pos)
    plt.figure(1,figsize=(12,12)) 
    ax = plt.gca()
    ax.set_axis_off()
    plt.show()

''' Example
G = genJohnsonGraph(5,2,0)
#drawGraph(G,layout='spectral')
#drawGraph(G,layout='spring')
#drawGraph(G,layout='shell',layout_array=[['45'],['12', '13', '23'],[ '14', '25', '34', '15', '24', '35']]) # Broken
'''

def sliceIdx(mode,maxval,s=0): # Generates indices for indexing slice of 3d array
    if mode == 'johnson': # Diagonal slice for Johnson graphs
        V,K =  np.meshgrid(range(maxval),range(maxval),indexing='ij')
        return V,K,K-1
    elif mode == 'kneser': # Really, this give v,k,i for fixed i...but i defaults to 0 (i.e. Kneser graphs)
        V,K =  np.meshgrid(range(maxval),range(maxval),indexing='ij')
        return V,K,s*np.ones((maxval,maxval),dtype=dtype)
    elif mode == 'fixedv':
        K,I =  np.meshgrid(range(maxval),range(maxval),indexing='ij')
        return maxval*np.ones((maxval,maxval),dtype=dtype),K,I

