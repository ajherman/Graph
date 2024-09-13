from scipy.optimize import linprog
from math import comb
import matplotlib.pyplot as plt
import numpy as np

n = 4 # Clique size
q = 0.5 # Overlap
symmetric = False

if symmetric:
    ##############################################
    # LP formulation for symmetric case (original)
    ##############################################
    A=np.array([comb(n-1,k) for k in range(n)])
    B=np.array([comb(n-2,k) for k in range(n)])
    A_eq = np.vstack([A,B])
    b_eq = np.array([1,q])
    c = np.array([comb(n,k) for k in range(n)])

    #################
    # Alternate basis
    #################
    A_alt = np.array([comb(n-2,k+1) for k in range(n-2)])
    B_alt = np.array([comb(n-2,k) for k in range(n-2)])
    A_ub = np.vstack([A_alt,B_alt])
    b_ub = np.array([1-q,q])
    c_alt = np.array([-(k+1)*comb(n-1,k+2) for k in range(n-2)])

    ##########
    # Solve LP
    ##########
    Q = np.linspace(0,1,200)
    SOLNS = []
    SOLNS_VEC = []
    
    T = np.eye(n-2)
    T_top = np.array([-comb(n-2,k+1) for k in range(n-2)])
    T_bottom = np.array([-comb(n-2,k) for k in range(n-2)])
    T = np.vstack([T_top,T,T_bottom])
    # print(T)
    # assert(0)

    for q in Q:
        # V1
        res = linprog(c, A_eq=A_eq, b_eq=[1,q], bounds=[0,1], method='highs')
        SOLNS.append(res.fun)
        SOLNS_VEC.append(res.x*c)

        # V2

        # Base flower vector
        flower = np.zeros(n)
        flower[0] = 1-q 
        flower[-1] = q

        res_alt = linprog(c_alt, A_ub=A_ub, b_ub=[1-q,q], bounds=[0,1], method='highs')
        # print('\nq:',q)
        # print(np.flip(res.x))
        # print(flower+T@res_alt.x)
        
        # If this version does not match the original, catch the mistake
        diff = np.flip(res.x) - (flower+T@res_alt.x)
        if np.max(np.abs(diff))>1e-10:
            print("LP solutions do not match")
            print(q)
            assert(0)

    offset = 1e-1
    SOLNS_VEC = np.array(SOLNS_VEC).T+offset
    SOLNS_VEC = np.cumsum(SOLNS_VEC,axis=0)

    print(SOLNS_VEC.shape)

    for i in range(n):
        plt.plot(Q,SOLNS_VEC[i],label=str(n-i))
    plt.plot(Q,n/(Q*(n-1)+1),label='n/(q(n-1)+1)',color='black',linestyle='--')
    plt.legend()

    # Add vertical lines at specific intervals
    tic_pos = np.arange(0,1,1/(n-1))
    for pos in tic_pos:
        plt.axvline(x=pos)
    plt.xticks(tic_pos)

    plt.show()

else:
    ##################################
    # Mistake in implementation below?
    ##################################

    # Define constraints

    # Bit representations of each region (also used to enforce that each vertex region should sum to 1)
    V=np.array([[int(x) for x in format(i, '0{}b'.format(n))] for i in range(1,2**n)]).T

    # print(V.shape)
    # assert(0)

    # Regions contained in each pair of vertices should sum to q
    Q=[]
    for i in range(n):
        for j in range(i):
            Q.append(V[i]&V[j])
    Q=np.array(Q)
    # Bit representations of each region (also used to enforce that each vertex region should sum to 1)
    V=np.array([[int(x) for x in format(i, '0{}b'.format(n))] for i in range(1,2**n)]).T

    # print(V.shape)
    # assert(0)

    # Regions contained in each pair of vertices should sum to q
    Q=[]
    for i in range(n):
        for j in range(i):
            Q.append(V[i]&V[j])
    Q=np.array(Q)

    A_eq = np.vstack([V,Q])

    # Right-hand side of equality constraints
    b_eq = np.ones(n*(n+1)//2)
    b_eq[n:] = q

    # print(A_eq)
    # print(b_eq)

    # Maximum sum of regions
    c = np.ones(2**n-1)
    A_eq = np.vstack([V,Q])

    # Right-hand side of equality constraints
    b_eq = np.ones(n*(n+1)//2)
    b_eq[n:] = q

    # print(A_eq)
    # print(b_eq)

    # Maximum sum of regions
    c = np.ones(2**n-1)

    # Solve LP
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[0,1], method='highs')

    # Print solution
    # print(res.x)
    # print(np.where(res.x>1e-10))

    by_int_type = [[] for i in range(n)]
    for i in range(2**n-1):
        num_ones = int(np.sum(V[:,i]))
        # print(num_ones)
        val = res.x[i]
        # print(val)
        by_int_type[num_ones-1].append(val)

        if np.abs(val)>0.01:
            print(f"{(i+1):0{n}b}")  # Print i in binary with length

    for j in range(n):
        print(f"Number of vertices: {j+1}")
        print(by_int_type[j])

###################################################################################
###################################################################################

