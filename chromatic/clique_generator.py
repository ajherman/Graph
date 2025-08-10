import numpy as np
from sympy import factorint
from itertools import combinations_with_replacement as cwr
from ortools.sat.python import cp_model
import math
import time
import argparse
import os
import json
import pprint

np.set_printoptions(threshold=np.inf)

# from mpi4py import MPI

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Enumerate all cliques in a graph.")
parser.add_argument("--n", type=int, default=7, help="Clique size")
parser.add_argument("--k", type=int, default=7, help="Vertex size")
parser.add_argument("--i", type=int, default=3, help="Vertex size")
parser.add_argument("--find_all", action="store_true", help="Find all cliques")
parser.add_argument("--find_min_v", action="store_true", help="Find minimum size of clique")
parser.add_argument("--find_bounds", action="store_true", help="Compute and store various bounds")
parser.add_argument("--time_limit", type=int, default=1e20, help="Time limit")

args = parser.parse_args()

time_limit = args.time_limit
n,k,i = args.n,args.k,args.i
find_all = args.find_all
find_min_v = args.find_min_v
find_bounds = args.find_bounds

print(f"\nWorking on n={n:02d}, k={k:02d}, i={i:02d}")

npy_file = f"cliques/{k},{i}_{n}.npy"
txt_file = f"cliques/{k},{i}_{n}.txt"

inf = int(10e10)
deza_flower_bound = max(i+2,(k-i)**2+(k-i)+1)
flower_size = n*(k-i)+i

if not os.path.exists('clique_data.json'):
    data = {}
else:
    print("Loading data from clique_data.json")
    with open('clique_data.json') as f:
        # Load json converting integer keys to integers and leaving other keys as strings
        data = json.load(f, object_hook=lambda d: {int(k) if k.lstrip('-').isdigit() else k: v for k, v in d.items()})

# if k in data and i in data[k] and n in data[k][i]:
#     if data[k][i][n]['min_v'] and data[k][i][n]['profiles'] != [] and data[k][i][n]['factorizations'] != []:
#         print("Already computed.")
#         exit()

if not k in data:
    data[k] = {}
if not i in data[k]:
    data[k][i] = {'info': {'omega*':None,'omega*_ub':deza_flower_bound,'omega*_lb':0}}
# Check if we have already computed this n
if not n in data[k][i]:
    data[k][i][n] = {'min_v':None,
                        'solutions':[],
                        'all solutions found':False,
                        'profiles':[],
                        'compute_time':0.0,
                        # 'counter_examples':None,
                        'is_flower':None,
                        # 'disaster':None,
                        'vector_space':None,
                        'irreducible':None,
                        'factorizations':[],

                        'cont_lb':None,
                        'flower_lb':None,
                        # 'flower_ub':n*(k-i)+i,
                        # 'star_ub':None,
                        'vector_ub':None,
                        'comp_ub':None,
                        # 'above_flower_lb':None,
                        'diff_lb':None,
                        'sum_ub':None}

if data[k][i][n]['min_v'] is not None and data[k][i][n]['factorizations'] != []:
    print("Already computed.")
    exit()

########################################
########################################
# Check if this is an automatic flower #
########################################
########################################

# # Flower lower bound (sufficiently large n forces clique to be a flower)
# if n > deza_flower_bound:
#     data[k][i][n]['is_flower'] = True
# elif n < 3:
#     data[k][i][n]['is_flower'] = True
# elif i == 0 or i == k:
#     data[k][i][n]['is_flower'] = True

# elif any([data[k][i][m]['is_flower'] for m in data[k][i] if type(m) is int and 3<=m<n]):
#     # If there is a lower n that is a flower, then must be  flower
#     data[k][i][n]['is_flower'] = True

# # If flower, no need to compute anything else
# if data[k][i][n]['is_flower']:
#     print("This is a flower.")
#     data[k][i][n]['min_v'] = n*(k-i)+i
#     data[k][i][n]['irreducible'] = k==1
#     data[k][i][n]['factorizations'] = [{f'({n},1,0)':k-i,'(1,1,1)':i}]
#     data[k][i][n]['all solutions found'] = True
#     data[k][i][n]['profiles'] = ['flower']
#     data[k][i][n]['time'] = 0.0
#     data[k][i][n]['sum_ub'] = n*(k-i)+i

#     # Save json data
#     with open("clique_data.json", "w") as f:
#         json.dump(data, f, ensure_ascii=False, separators=(',', ': '), indent=4)

#     # Save solutions
#     solutions = (k-i)*np.ones((1,2**n-1),dtype=int)
#     solutions[0,-1] = i
#     np.savez_compressed(npy_file, np.zeros((0,n),dtype=int))
#     data[k][i][n]['all solutions found'] = True

#     with open(txt_file, "a") as f:
#         f.write(f"\n\nn={n},k={k},i={i}: min v = {n*(k-i)+i}")
#         f.write(data[k][i][n]['profiles'][0])

#     exit()
                

##################
##################
# Help functions #
##################
##################

# Function to check if n is a prime power
def is_prime_power(n):
    if n < 2:
        return False
    
    # Factorize the number
    factors = factorint(n)
    
    # A prime power has exactly one prime factor
    if len(factors) == 1:
        prime, exponent = list(factors.items())[0]
        return exponent >= 1  # Exponent should be at least 1 for prime power
    
    return False


def solve_min_v(A, b, C=None, d=None, lb=None, ub=None):
    # Initialize the constraint programming model
    model = cp_model.CpModel()

    num_rows = len(A)
    num_vars = len(A[0])
    
    # x = [model.NewIntVar(0, b[0]-b[-1], f'x[{i}]') for i in range(n)] + \
    #     [model.NewIntVar(0, b[-1],f'x[{i}]') for i in range(n,num_vars)]
    
    x = [model.NewIntVar(0, b[0], f'x[{i}]') for i in range(num_vars)]
    
    # print(A)
    # print(b)
    # # print(C)
    # # print(d)
    # print(x)
    # # print(lb)
    # # print(ub)
    # # assert(0)

    # Add constraints Ax = b
    for i in range(num_rows):
        model.Add(sum(A[i][j] * x[j] for j in range(num_vars)) == b[i])

    if not C is None:
        for i in range(n-1):
            model.Add(sum(C[i][j] * x[j] for j in range(num_vars)) >= d[i])

    if not lb is None:
        model.Add(sum(x) >= lb)

    if not ub is None:
        model.Add(sum(x) <= ub)

    if lb == ub:
        print("Lower and upper bounds are equal")
        return lb

    # Objective: minimize the sum of x
    model.Minimize(sum(x))

    # Create the solver
    solver = cp_model.CpSolver()

    # Solve the problem to get the minimum sum
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        min_v = sum(solver.Value(x[i]) for i in range(num_vars))
        return min_v
    else:
        return None

def find_all_solutions_with_min_v(A, b, min_v, C=None, d=None, time_limit=1e10, old_solutions=[]):
    # Initialize the constraint programming model
    model = cp_model.CpModel()

    num_rows = len(A)
    num_vars = len(A[0])

    x = [model.NewIntVar(0, b[0]-b[-1], f'x[{i}]') for i in range(n)] + \
        [model.NewIntVar(0, b[-1],f'x[{i}]') for i in range(n,num_vars)]

    # Add constraints Ax = b
    for i in range(num_rows):
        model.Add(sum(A[i][j] * x[j] for j in range(num_vars)) == b[i])

    if not C is None:
        for i in range(n-1):
            model.Add(sum(C[i][j] * x[j] for j in range(num_vars)) >= d[i])

    # Add constraint that the sum of x equals the minimum sum
    model.Add(sum(x) == min_v)

    # Create the solver
    solver = cp_model.CpSolver()

    # Collect all solutions
    solutions = []

    # Define a solution callback to collect solutions
    class SolutionCollector(cp_model.CpSolverSolutionCallback):
        def __init__(self, variables, time_limit, old_solutions=[]):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self._variables = variables
            self._solutions = old_solutions
            self.creation_time = time.time()
            self.time_limit = time_limit

        def on_solution_callback(self):
            solution = [self.Value(var) for var in self._variables]
            if solution not in self._solutions:
                self._solutions.append(solution)
            if time.time()-self.creation_time>self.time_limit:
                self.StopSearch()

        def get_solutions(self):
            return np.array(self._solutions)

    # Create the solution collector
    collector = SolutionCollector(x,time_limit=time_limit,old_solutions=old_solutions)

    # Search for all solutions
    solver.SearchForAllSolutions(model, collector)

    # Return the list of all solutions
    return collector.get_solutions()

#############
#############
# Main loop #
#############
#############

tic = time.time()

##################################
##################################
# Compute upper and lower bounds #
##################################
##################################

# Lower bound from continuous min (this is always a lower bound)
if (n-1)*i%k == 0:
    q = (n-1)*i//k + 1
    if n*k%q == 0:
        cont_lb = n*k//q
    else:
        cont_lb = math.ceil(n*k/q)
else:
    q = (n-1)*i/k+1
    cont_lb = n*k*(1+(np.ceil(q)-q)/np.floor(q))/np.ceil(q) 
    if cont_lb-math.floor(cont_lb) < 1e-8:
        cont_lb = math.floor(cont_lb)
    else:
        cont_lb = math.ceil(cont_lb)

# TODO: Add more bounds here
# Vector space upper bound (if there is a vector space clique, this gives an ub)
if is_prime_power(k-1) and i == 1 and n<=k**2-k+1:
    vector_ub = k**2-k+1
else:
    vector_ub = inf

#############################
#############################
# Bounds from other cliques #
#############################
#############################

# min v is non-decreasing in n
nondec_lb = max([data[k][i][m]['min_v'] for m in data[k][i] if type(m) is int and m<n], default=0)
nondec_ub = min([data[k][i][m]['min_v'] for m in data[k][i] if type(m) is int and m>n], default=inf)

# Above flower bound
above_flower_lb = 0
if n > 3:
    try:
        if data[k][i][n-1]['is_flower']:
            above_flower_lb = n*(k-i)+i
    except:
        pass

# Sum and different bounds
diff_lb, sum_ub = 0,inf
comp_lb, comp_ub = 0, inf

factorizations = []
for k1 in data:
    if type(k1) is int:
        if k1 < k: # Sums
            for i1 in range(k1+1):
                if k1 - k + i <= i1 <= i:
                    if not i1 in data[k1] or not n in data[k1][i1]:
                        raise Exception(f"Error: n={n}, k={k}, i={i}, k1={k1}, i1={i1}")
    
                    k2,i2 = k-k1, i-i1
                    if k2 - k + i <= i2 <= i:
                        if not k2 in data or not i2 in data[k2] or not n in data[k2][i2]:
                            raise Exception(f"Error: n={n}, k={k}, i={i}, k1={k1}, i1={i1}, k2={k2}, i2={i2}")
                    try:
                        v1 = data[k1][i1][n]['min_v']
                        v2 = data[k2][i2][n]['min_v']
                    except:
                        raise Exception(f"Error: n={n}, k={k}, i={i}, k1={k1}, i1={i1}, k2={k2}, i2={i2}")
                  
                    v = v1+v2
                    if v <= sum_ub:
                        if v < sum_ub:                        
                            sum_ub = v
                            factorizations = []
                        for x in data[k1][i1][n]['factorizations']:
                            for y in data[k2][i2][n]['factorizations']:
                                factorization = {}
                                components = np.unique(list(x)+list(y))
                                for z in components:
                                    coef = 0
                                    if z in x:
                                        coef += x[z]
                                    if z in y:
                                        coef += y[z]
                                    if coef > 0:
                                        factorization[z] = coef
                                if not factorization in factorizations:
                                    factorizations.append(factorization)

        elif k1 > k: # Differences
            for i1 in range(k1+1):
                if k1 - k + i >= i1 >= i:
                    if k1 - k in data and i1 - i in data[k1-k] and k1 in data and i1 in data[k1] and n in data[k1][i1] and n in data[k1-k][i1-i]: 
                        k2, i2 = k1-k, i1-i
                        try:
                            v1 = data[k1][i1][n]['min_v']
                            v2 = data[k2][i2][n]['min_v']
                        except:
                            raise Exception(f"Error: n={n}, k={k}, i={i}, k1={k1}, i1={i1}, k2={k2}, i2={i2}")
                        v = v1-v2
                        diff_lb = max(diff_lb, v)

        if k1 != k: # Complements
            i1 = k1-k+i
            if k1 >= i1 >=0: 
                if k1 in data and i1 in data[k1] and n in data[k1][i1]:
                    v1 = data[k1][i1][n]['min_v']
                    if v1 <= k + k1:
                        comp_ub = min(comp_ub, k+k1)
                    if v1 > k + k1:
                        comp_lb = max(comp_lb, k+k1+1)

data[k][i][n]['cont_lb'] = cont_lb
data[k][i][n]['nondec_lb'] = nondec_lb
data[k][i][n]['nondec_ub'] = nondec_ub
data[k][i][n]['diff_lb'] = diff_lb
data[k][i][n]['sum_ub'] = sum_ub
data[k][i][n]['vector_ub'] = vector_ub
data[k][i][n]['comp_ub'] = comp_ub

lb = max(cont_lb, nondec_lb, comp_lb, diff_lb, above_flower_lb)
ub = min(nondec_ub, comp_ub, sum_ub, vector_ub)

if ub<lb: # If this is not true, we have a problem
    print("Lower bound:", lb)
    print("Upper bound:", ub)
    assert(0)


################################
################################
# Step 1: Find the minimum sum #
################################
################################

if find_min_v:

    # Set up the constraints
    V=np.array([[int(x) for x in format(j, '0{}b'.format(n))] for j in range(1,2**n)]).T

    # Vertices must have size k
    A_k = V
    b_k = k*np.ones(n, dtype=int)

    # Intersections of two vertices must have size i
    A_i=np.zeros((0,2**n-1),dtype=int)
    for j1 in range(n):
        for j2 in range(j1):
            A_i = np.vstack([A_i, V[j1]&V[j2]])

    b_i = i*np.ones(len(A_i), dtype=int)

    # Equality constraints
    A = np.vstack([A_k,A_i])
    b = np.hstack([b_k,b_i])

    # Inequality constraints

    # Symmetry breaking constraint
    SB_C = np.zeros((n-1,2**n-1),dtype=int)
    for j in range(n-1):
        SB_C[j] = V[j+1]-V[j]
    SB_C = SB_C*np.power(4,np.sum(V,axis=0,keepdims=True))
    SB_d = np.zeros(n-1,dtype=int)

    # From Lint theorem tau*t*(n-t) >= n*(k-i) for all t

    t = np.sum(V, axis=0)
    levels = np.arange(1,n)
    LINT_C = t.reshape(1,-1) == levels.reshape(-1,1)
    LINT_C = LINT_C.astype(int)
    LINT_d = (n*(k-i)//(levels*(n-levels)))

    C = np.vstack([SB_C,LINT_C])
    d = np.hstack([SB_d,LINT_d])

    # Reorder the columns of A, V, Q, and SB
    idx=np.argsort(t,kind='stable')
    splits = [math.comb(n, j) for j in range(1,n+1)]
    splits = np.cumsum(splits)
    splits = [0] + splits.tolist()
    V = V[:,idx]
    A = A[:,idx]
    C = C[:,idx]

    A_i = A_i[:,idx]
    A_k = A_k[:,idx]
    SB_C = SB_C[:,idx]
    LINT_C = LINT_C[:,idx]

    # Find the minimum sum
    min_v = solve_min_v(A, b, C=C, d=d, lb=lb, ub=ub)
    data[k][i][n]['min_v'] = min_v

    # assert(min_v is not None)
    if min_v is None:
        print("No solution found.")
        exit()

    print(f"Minimum sum: {min_v}")


    # Check if the minimum sum achieves any of the bounds
    if min_v == cont_lb:
        print("Continuous lower bound achieved.")

    if min_v == n*(k-i)+i:
        print("This is a flower.")
        data[k][i][n]['is_flower'] = True
    elif min_v < n*(k-i)+i:
        data[k][i][n]['is_flower'] = False

    if min_v == data[k][i][n]['sum_ub']:
        data[k][i][n]['irreducible'] = False
    elif min_v < data[k][i][n]['sum_ub']:
        print("This is irreducible.")
        data[k][i][n]['irreducible'] = True
        factorizations = [{f'({min_v},{k},{i})':1}]

    data[k][i][n]['factorizations'] = factorizations

    if min_v == data[k][i][n]['vector_ub']:
        print("Vector space upper bound achieved.")
        data[k][i][n]['vector_space'] = True
    else:
        data[k][i][n]['vector_space'] = False
        
    with open("clique_data.json", "w") as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ': '), indent=4)

if find_all:
    #################################################
    # Step 2: Find all solutions with the minimum sum
    #################################################
    if data[k][i][n]['all solutions found']:
        pass
    elif data[k][i][n]['min_v'] is None:
        raise Exception("Must compute min sum before searching for solutions")
    elif data[k][i][n]['flower']:
        solutions = (k-i)*np.ones((1,2**n-1),dtype=int)
        solutions[0,-1] = i
        np.savez_compressed(npy_file, np.zeros((0,n),dtype=int))
        data[k][i][n]['all solutions found'] = True
    else:
        solutions = find_all_solutions_with_min_v(A, b, min_v, C=C, d=d, time_limit=time_limit, old_solutions=[])
        np.savez_compressed(npy_file, solutions)
        data[k][i][n]['all solutions found'] = True

    ###################
    # Analyze solutions
    ###################
    if data[k][i][n]['all solutions found'] and data[k][i][n]['profiles'] is None:

        profiles = []

        for solution in solutions:
            # counter_example = False

            profile_str = ''
            for j in range(len(splits)-1):
                region = solution[splits[j]:splits[j+1]]
                unique_vals = np.unique(region)
                avg_val = np.mean(region)
                # if 0 in unique_vals and len(unique_vals)==1:
                #     continue
                # else:
                #     if not np.floor(q)<=j<=np.ceil(q):
                #         counter_example = True

                vertex_sizes=np.unique(V[:,splits[j]:splits[j+1]]@region)
                inter_sizes = np.unique(Q[:,splits[j]:splits[j+1]]@region)

                profile_str += f"| <{j+1}> vals:{unique_vals}, total:{np.sum(region)}, v sizes:{vertex_sizes}, i sizes:{inter_sizes} |"
            # data[k][i][n]['counter_examples'] = counter_example
            
            profile_str += "\n"
            profiles.append(profile_str)

        for profile_str in set(profiles):
            data[k][i][n]['profiles'].append(profile_str)

    with open(txt_file, "a") as f:
        f.write(f"\n\nn={n},k={k},i={i}: min v = {min_v}")
    for j in range(len(splits)-1):
        region = solutions[:,splits[j]:splits[j+1]]
        if np.sum(region)>0:
            with open(txt_file, "a") as f:
                f.write(f"\n\n{j+1}-intersecting regions:\n")
                np.savetxt(f, region, fmt='%d')

# Get total elapsed time          
toc = time.time()
elapsed_time = toc-tic
data[k][i][n]['time'] = elapsed_time

# Save data to json
with open('clique_data.json', 'w') as f:
    json.dump(data, f, ensure_ascii=False, separators=(',', ': '), indent=4)

print("Elapsed time:", elapsed_time, "seconds.")