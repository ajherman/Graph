import numpy as np
import itertools as it

d = 3

def B(v,d): # Generates a set of vectors that should not be in the nullspace of M
    perms=list(it.permutations(range(v),2*d))
    X = np.zeros((len(perms),v),dtype=np.int16)
    for idx1,idx2 in enumerate(perms):
        X[idx1,idx2[:d]]=1
        X[idx1,idx2[d:]]=-1
    return X

def M(v,d): # Set of vectors using idea of consecutive powers
    y = np.arange(v,dtype=np.int16)
    return np.array([y**i for i in range(1,d+1)]).T%v

def N(v,d): # Set of vectors using idea of consecutive powers
    y = np.arange(v,dtype=np.int16)
    return np.array([y**i for i in range(1,d)]).T%v

def test(v,d): # Returns True if the columns of M have the property that no two disjoint d-sets have the same sum
    C = np.dot(B(v,d),M(v,d))
    return np.min(np.sum(np.abs(C),axis=1))>0

def isPrime(x):
    return 2 in [x,2**x%x]

for i in range(2*d+1,100):
    if isPrime(i):
        print(test(i,d))
