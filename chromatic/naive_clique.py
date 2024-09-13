from scipy.optimize import linprog
from math import comb
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import time

n = 5 # Clique size
k,i = 12,3


v = (k-i)*n+i # Flower
# Iterate over all 2^n-1 tuples with non-negative integer values summing to at most v
    # Bit representations of each region (also used to enforce that each vertex region should sum to 1)
V=np.array([[int(x) for x in format(j, '0{}b'.format(n))] for j in range(1,2**n)]).T

# print(V.shape)
# assert(0)

# Regions contained in each pair of vertices should sum to q
Q=[]
for j1 in range(n):
    for j2 in range(j1):
        Q.append(V[j1]&V[j2])
Q=np.array(Q)

A_eq = np.vstack([V,Q])

# Right-hand side of equality constraints
b_eq = k*np.ones(n*(n+1)//2)
b_eq[n:] = i

one_or_two_idx = [j for j in range (2**n-1) if sum(V[:n,j]) == 1 or sum(V[:n,j]) == 2]
more_than_two_idx = [j for j in range (2**n-1) if sum(V[:n,j]) > 2]
# print(one_or_two_idx)

# print(A_eq)

# Precompute relevant matrices
R = A_eq[:,one_or_two_idx]
R_inv = np.linalg.inv(R)
S = A_eq[:,more_than_two_idx]
c = np.ones((1,n*(n+1)//2))@R_inv@S - 1
d_eq= R_inv@b_eq
M = R_inv@S

# print("c: ", c)
# print("M.shape: ", M.shape)
# print(np.mean(M==-1))
# assert(0)

# Iterate over all 2^n-1 tuples with non-negative integer values summing to at most v
import itertools
import numpy as np

# Define the range for the tuples
value_range = range(i)

# Generate all tuples of length 2^n-1 with values in the specified range
tuples = itertools.product(value_range, repeat=2**n-1-n*(n+1)//2)

best_val = np.inf
best_z = None
max_c = -np.inf
total_combinations = i**(2**n-1-n*(n+1)//2)
print("Total number of tests: ", total_combinations)
# progress_bar = tqdm(total=total_combinations, desc="Progress", unit="combination")


# def generate_k_tuples(i, k, batch_size=1000):
#     value_range = range(i + 1)  # Elements from 0 to i
#     tuples = itertools.product(value_range, repeat=k)
#     return np.array(list(tuples))

# X = generate_k_tuples(i, 2**n-1-n*(n+1)//2)


tic = time.time()
batch_size=1000000
X = np.zeros((2**n-1-n*(n+1)//2,batch_size))
batch_num = 0
n_batches = total_combinations//batch_size
while any(tuples):
    tic = time.time()
    for j in range(batch_size):
        X[:,j] = next(tuples)
    y = d_eq.reshape((-1,1)) - M @ X
    # print(np.max(y))

    
    cond1 = c.reshape((1,-1))@X>max_c
    cond2 = np.all(y>=0, axis=0)

    print(np.sum(cond1))
    print(np.sum(cond2))
    
    # print(cond1.shape)
    # print(cond2.shape)
    idx = np.where(cond1 & cond2)

    for j in idx[1]:    
        # progress_bar.update(1)
        x = X[:,j]
        # y = R_inv @ (b_eq - S @ x)
        # val = np.dot(c,x)
        y = d_eq - M @ x

        if np.dot(c,x)>max_c: # First check if this is better than other cliques
            if np.all(y >= 0): # Then check if all regions are non-negative
                max_c = np.dot(c,x)
                z = np.zeros(2**n-1)
                z[more_than_two_idx] = x
                z[one_or_two_idx] = y
                best_val = np.sum(z)
                best_z = z

                print("Best value: ", best_val) 
                print("Should be non-negative: ", z)
                # print("Should be 0: ", A_eq @ z - b_eq)
                print("")

    toc = time.time()
    print("Elapsed time for batch ",batch_num," / ", n_batches, ": ", toc-tic," seconds")
    batch_num += 1

# toc = time.time()
# print("Elapsed time: ", toc-tic)


"""
for j,x in enumerate(tuples):

    # progress_bar.update(1)
    x = np.array(x)
    # y = R_inv @ (b_eq - S @ x)
    # val = np.dot(c,x)
    y = d_eq - M @ x

    # assert(np.sum(z)==np.sum(d_eq)-np.dot(c,x))
    # assert((np.sum(z)<best_val) == (np.dot(c,x)>max_c))

    if j % (total_combinations//100) == 0:
        print("Progress: ", j, "/", total_combinations, "[", (100*j)//total_combinations, "%]")
    if np.all(y >= 0): # Then check if all regions are non-negative
        if np.dot(c,x)>max_c: # First check if this is better than other cliques
            max_c = np.dot(c,x)
            z = np.zeros(2**n-1)
            z[more_than_two_idx] = x
            z[one_or_two_idx] = y
            best_val = np.sum(z)
            best_z = z

            print("Best value: ", best_val) 
            print("Should be non-negative: ", z)
            # print("Should be 0: ", A_eq @ z - b_eq)
            print("")
# progress_bar.close()
    



##########################################################

# import itertools

# def generate_combinations(n):
#     all_combinations = []
#     for length in range(1, n + 1):
#         combinations = list(itertools.combinations(range(n), length))
#         all_combinations.append(sorted(combinations))
#     return all_combinations

# n = 4  # Change this to the desired value of n
# combinations = generate_combinations(n)

# for length, group in enumerate(combinations, start=1):
#     print(f"Length {length}:")
#     for combo in group:
#         print(combo)

"""