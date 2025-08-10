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
parser.add_argument("--find_min_sum", action="store_true", help="Find minimum size of clique")
parser.add_argument("--find_bounds", action="store_true", help="Compute and store various bounds")
parser.add_argument("--time_limit", type=int, default=1e10, help="Time limit")

args = parser.parse_args()

time_limit = args.time_limit
n,k,i = args.n,args.k,args.i
find_all = args.find_all
find_min_sum = args.find_min_sum
find_bounds = args.find_bounds

print(f"\nWorking on n={n:02d}, k={k:02d}, i={i:02d}")

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


def solve_min_sum(A, b, C=None, d=None, lb=None, ub=None):
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
        min_sum = sum(solver.Value(x[i]) for i in range(num_vars))
        return min_sum
    else:
        return None

def find_all_solutions_with_min_sum(A, b, min_sum, C=None, d=None, time_limit=1e10, old_solutions=[]):
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
    model.Add(sum(x) == min_sum)

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

if not os.path.exists('clique_data.json'):
    data = {}
else:
    print("Loading data from clique_data.json")
    with open('clique_data.json') as f:
        data = json.load(f)

# Main loop
os.makedirs('new_cliques', exist_ok=True)

tic = time.time()

# Check if we have already started computing this k,i
ki = str(k)+','+str(i)
nn = str(n)

if not ki in data:
    data[ki] = {'omega star':None,
                        'omega star ub':(k-i)*math.comb(k,i)+1,
                        'omega star lb':0}
    
# Check if we have already computed this n
if not nn in data[ki]:
    data[ki][nn] = {'continuous_min':None,
                        'min_sum':None,
                        'solutions':[],
                        'all solutions found':False,
                        'profiles':None,
                        'time':0.0,
                        'counter_examples':None,
                        'flower':None,
                        'disaster':None,
                        'vector_space':None,
                        'complement':None,
                        'prime':None,

                        'cont_lb':None,
                        'flower_lb':None,
                        'flower_ub':n*(k-i)+i,
                        'star_ub':None,
                        'vector_ub':None,
                        'comp_ub':None,
                        'above_flower_lb':None,
                        'diff_lb':None,
                        'sum_ub':None}


if find_bounds:

    flower_size = (k-i)*n+i # Size of a flower

    ################################
    # Compute upper and lower bounds
    ################################

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

    # Flower upper bound (this is always an upper bound)
    flower_ub = (k-i)*n+i

    # Flower lower bound (sufficiently large n forces clique to be a flower)
    if n > (k-i)*math.comb(k,i)+1:
        flower_lb = (k-i)*n+i
    else:
        flower_lb = 0

    # TODO: Add more bounds here
    # Vector space upper bound (if there is a vector space clique, this gives an ub)
    if is_prime_power(k-1) and i == 1 and n<=k**2-k+1:
        vector_ub = k**2-k+1
    else:
        vector_ub = 10e10

    ###########################
    # Bounds from other cliques
    ###########################

    # Above a flower (if there is a lower n that is a flower, then must be  flower)
    above_flower_lb = 0
    nondec_lb, nondec_ub = 0, n*(k-i)+i
    max_n = max([int(x) for x in data[ki].keys() if type(data[ki][x]) is dict])

    for m in range(3,max_n):
        mm = str(m)
        if mm in data[ki]:
            if m<n:
                # Nondecreasing in n
                if data[ki][mm]['min_sum'] is not None:
                    nondec_lb = max(nondec_lb, data[ki][mm]['min_sum'])

                # If any lower n is a flower, then must be a flower
                if data[ki][mm]['min_sum']==m*(k-i)+i or data[ki][mm]['flower']:
                    above_flower_lb = n*(k-i)+i
            elif m>n:
                # Nondecreasing in n
                if data[ki][mm]['min_sum'] is not None:
                    nondec_ub = min(nondec_ub, data[ki][mm]['min_sum'])

    # Sum and different bounds
    diff_lb, sum_ub = 0, 10e10
    comp_ub = 10e10
    comp_k,comp_i=None,None
    for ki2 in data:
        if nn in data[ki2] and data[ki2][nn]['min_sum'] is not None:
            m2 = data[ki2][nn]['min_sum']
            k2,i2 = [int(x) for x in ki2.split(',')]

            # Complement upper bound
            if k2 != k and k2 - i2 == k - i and data[ki2][nn]['min_sum'] is not None:
                if m2 <= k+k2:
                    if k+k2 < comp_ub:
                        comp_ub = k+k2
                        comp_k,comp_i = k2,i2
                        data[ki][nn]['comps'] = (comp_ub,comp_k,comp_i)

            for ki3 in data:
                if nn in data[ki3] and data[ki3][nn]['min_sum'] is not None:
                    m3 = data[ki3][nn]['min_sum']
                    k3,i3 = [int(x) for x in ki3.split(',')]

                    # Sum upper bound
                    if k == k2+k3 and i == i2+i3 and k2 != 0:
                        sum_ub = min(sum_ub, m2+m3)

                    # Difference lower bound
                    if k3 == k + k2 and i3 == i + i2 and k2 != 0:
                        diff_lb = max(diff_lb, m3-m2)

    data[ki][nn]['cont_lb'] = cont_lb
    data[ki][nn]['flower_ub'] = flower_ub
    data[ki][nn]['flower_lb'] = flower_lb
    data[ki][nn]['above_flower_lb'] = above_flower_lb
    data[ki][nn]['nondec_lb'] = nondec_lb
    data[ki][nn]['nondec_ub'] = nondec_ub
    data[ki][nn]['diff_lb'] = diff_lb
    data[ki][nn]['sum_ub'] = sum_ub
    data[ki][nn]['vector_ub'] = vector_ub
    data[ki][nn]['comp_ub'] = comp_ub

    lb = max(cont_lb, flower_lb, nondec_lb, above_flower_lb, diff_lb)
    ub = min(flower_ub, nondec_ub, comp_ub, sum_ub, vector_ub)

    if ub<lb:
        print("Lower bound:", lb)
        print("Upper bound:", ub)
        assert(0)

    with open("clique_data.json", "w") as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ': '), indent=4)


    ##############################
    # Step 1: Find the minimum sum
    ##############################

if find_min_sum:

    if True: # data[ki][nn]['min_sum'] is None:
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
        min_sum = solve_min_sum(A, b, C=C, d=d, lb=lb, ub=ub)
        data[ki][nn]['min_sum'] = min_sum

    min_sum = data[ki][nn]['min_sum']
    print(f"Minimum sum: {min_sum}")

    # Check if the minimum sum achieves any of the bounds
    if min_sum == cont_lb:
        print("Continuous lower bound achieved.")
        data[ki][nn]['disaster'] = True
    else:
        data[ki][nn]['disaster'] = False

    if min_sum == flower_ub:
        print("This is a flower.")
        data[ki][nn]['flower'] = True
    else:
        data[ki][nn]['flower'] = False

    if min_sum == diff_lb:
        print("This is the difference of cliques.")

    if min_sum == sum_ub:
        print("The is the sum of cliques")
        data[ki][nn]['prime'] = False
    else:
        data[ki][nn]['prime'] = True

    if min_sum == vector_ub:
        print("Vector space upper bound achieved.")
        data[ki][nn]['vector_space'] = True
    else:
        data[ki][nn]['vector_space'] = False

    if min_sum == comp_ub:
        print("Complement upper bound achieved.")
        data[ki][nn]['complement'] = True
    else:
        data[ki][nn]['complement'] = False
        
    with open("clique_data.json", "w") as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ': '), indent=4)

if find_all:
    #################################################
    # Step 2: Find all solutions with the minimum sum
    #################################################
    if data[ki][nn]['all solutions found']:
        pass
    elif data[ki][nn]['min_sum'] is None:
        raise Exception("Must compute min sum before searching for solutions")
    elif data[ki][nn]['flower']:
        solutions = (k-i)*np.ones((1,2**n-1),dtype=int)
        solutions[0,-1] = i
        np.savez_compressed(f"new_cliques/{ki}_{n}.npy", np.zeros((0,n),dtype=int))
        data[ki][nn]['all solutions found'] = True
    else:
        solutions = find_all_solutions_with_min_sum(A, b, min_sum, C=C, d=d, time_limit=time_limit, old_solutions=[])
        np.savez_compressed(f"new_cliques/{ki}_{n}.npy", solutions)
        data[ki][nn]['all solutions found'] = True

    ###################
    # Analyze solutions
    ###################
    if data[ki][nn]['all solutions found'] and data[ki][nn]['profiles'] is None:

        profiles = []

        for solution in solutions:
            counter_example = False

            profile_str = ''
            for j in range(len(splits)-1):
                region = solution[splits[j]:splits[j+1]]
                unique_vals = np.unique(region)
                avg_val = np.mean(region)
                if 0 in unique_vals and len(unique_vals)==1:
                    continue
                else:
                    if not np.floor(q)<=j<=np.ceil(q):
                        counter_example = True

                    vertex_sizes=np.unique(V[:,splits[j]:splits[j+1]]@region)
                    inter_sizes = np.unique(Q[:,splits[j]:splits[j+1]]@region)

                    profile_str += f"| <{j+1}> vals:{unique_vals}, total:{np.sum(region)}, v sizes:{vertex_sizes}, i sizes:{inter_sizes} |"
            data[ki][nn]['counter_examples'] = counter_example
            
            profile_str += "\n"
            profiles.append(profile_str)

        for profile_str in set(profiles):
            data[ki][nn]['profiles'].append(profile_str)

    with open(f"new_cliques/{ki}_{n}.txt", "a") as f:
        f.write(f"\n\nn={n},k={k},i={i}: min v = {min_sum}")
    for j in range(len(splits)-1):
        region = solutions[:,splits[j]:splits[j+1]]
        if np.sum(region)>0:
            with open(f"new_cliques/{ki}_{n}.txt", "a") as f:
                f.write(f"\n\n{j+1}-intersecting regions:\n")
                np.savetxt(f, region, fmt='%d')

# Get total elapsed time          
toc = time.time()
elapsed_time = toc-tic
data[ki][nn]['time'] = elapsed_time

# Save data to json
with open('clique_data.json', 'w') as f:
    json.dump(data, f, ensure_ascii=False, separators=(',', ': '), indent=4)

print("Elapsed time:", elapsed_time, "seconds.")