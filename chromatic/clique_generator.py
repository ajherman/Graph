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
parser.add_argument("--time_limit", type=int, default=1e10, help="Time limit")

args = parser.parse_args()

time_limit = args.time_limit
n,k,i = args.n,args.k,args.i
find_all = args.find_all
find_min_sum = args.find_min_sum

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


def solve_min_sum(A, b, C=None, lb=None, ub=None):
    # Initialize the constraint programming model
    model = cp_model.CpModel()

    num_rows = len(A)
    num_vars = len(A[0])
    
    # Create non-negative integer variables
    # x = [model.NewIntVar(0, max(b[-1],b[0]-b[-1]), f'x[{j}]') for j in range(num_vars)]

    x = [model.NewIntVar(0, b[0]-b[-1], f'x[{i}]' for i in range(n))] + \
        [model.NewIntVar(0, b[-1],f'x[{i}]' for i in range(n,num_vars))]

    # Add constraints Ax = b
    for i in range(num_rows):
        model.Add(sum(A[i][j] * x[j] for j in range(num_vars)) == b[i])

    if not C is None:
        for i in range(n-1):
            model.Add(sum(C[i][j] * x[j] for j in range(num_vars)) >= 0)

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

def find_all_solutions_with_min_sum(A, b, min_sum, C=None, time_limit=1e10, old_solutions=[]):
    # Initialize the constraint programming model
    model = cp_model.CpModel()

    num_rows = len(A)
    num_vars = len(A[0])

    # Create non-negative integer variables
    # x = [model.NewIntVar(0, max(b[-1],b[0]-b[-1]), f'x[{i}]') for i in range(num_vars)]

    x = [model.NewIntVar(0, b[0]-b[-1], f'x[{i}]' for i in range(n))] + \
        [model.NewIntVar(0, b[-1],f'x[{i}]' for i in range(n,num_vars))]

    # Add constraints Ax = b
    for i in range(num_rows):
        model.Add(sum(A[i][j] * x[j] for j in range(num_vars)) == b[i])

    if not C is None:
        for i in range(n-1):
            model.Add(sum(C[i][j] * x[j] for j in range(num_vars)) >= 0)

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
                        'type':None}
 
# ############################################
# # Step 0: Check if all solutions are flowers
# ############################################
# flower_size = (k-i)*n+i # Size of a flower
# if data[ki][nn]['flower'] is None:
#     V=np.array([[int(x) for x in format(j, '0{}b'.format(n))] for j in range(1,2**n)]).T

#     # Regions contained in each pair of vertices should sum to q
#     Q=[]
#     for j1 in range(n):
#         for j2 in range(j1):
#             Q.append(V[j1]&V[j2])
#     Q=np.array(Q)
    
#     if n == 1:
#         Q = np.zeros((0,1),dtype=int)

#     # Symmetry breaking constraint
#     SB = np.zeros((n-1,2**n-1),dtype=int)
#     for j in range(n-1):
#         SB[j] = V[j+1]-V[j]
#     SB = SB*np.power(4,np.sum(V,axis=0,keepdims=True))

#     A = np.vstack([V,Q])

#     # Reorder the columns of A, V, Q, and SB
#     r = np.sum(V, axis=0)
#     idx=np.argsort(r,kind='stable')
#     splits = [math.comb(n, j) for j in range(1,n+1)]
#     splits = np.cumsum(splits)
#     splits = [0] + splits.tolist()
#     A = A[:,idx]
#     V = V[:,idx]
#     Q = Q[:,idx]
#     SB = SB[:,idx]

#     # Right-hand side of equality constraints
#     b = k*np.ones(n*(n+1)//2, dtype=int)
#     b[n:] = i

#     # Check if there are any cliques smaller than flowers
#     feasible = is_feasible(A, b, flower_size-1, C=SB)
#     if feasible:
#         data[ki][n]['flower']=False
#         data[ki][n]['type']='flower'
#     else:
#         data[ki][n]['flower']=True


##############################
# Start with easy computations
##############################
flower_size = (k-i)*n+i # Size of a flower

# Try to determine if this is a flower
if n > (k-i)*math.comb(k,i)+1 or i==0: # Flower special case
    data[ki][nn]['flower'] = True
    data[ki][nn]['type'] = 'flower'
    data[ki][nn]['min_sum'] = n*(k-i)+i
elif is_prime_power(k-1) and i == 1 and n<=(k-i)*math.comb(k,i)+1:
    data[ki][nn]['flower'] = False
    if n==k**2-k+1:
        data[ki]['min_sum']=k**2-k+1
elif data[ki][nn]['min_sum'] == n*(k-i)+i:
    data[ki][nn]['flower'] = True
    data[ki][nn]['type'] = 'flower'
elif data[ki][nn]['min_sum'] is not None and data[ki][nn]['min_sum']<n*(k-i)+i:
    data[ki][nn]['flower'] = False
elif data[ki][nn]['min_sum'] is None:
    for m in range(2,n):
        if data[ki][str(m)]['min_sum']==m*(k-i)+i or data[ki][str(m)]['flower']:
            data[ki][nn]['flower']=True
            data[ki]['type']='flower'
            data[ki]['min_sum']=n*(k-i)+i
            break
        
with open("clique_data.json", "w") as f:
    json.dump(data, f, ensure_ascii=False, separators=(',', ': '), indent=4)

if find_min_sum:
    ##############################
    # Step 1: Find the minimum sum
    ##############################
    flower_size = (k-i)*n+i # Size of a flower

    # Check if already computed
    if not data[ki][nn]['min_sum'] is None:
        print(f"Already computed: {n} vertices, {k} vertices, {i} intersections")
        print(data[ki][nn]['min_sum'])
    elif n > (k-i)*math.comb(k,i)+1 or i==0: # Flower special case
        # print("This is an automatic flower.")
        data[ki][nn]['min_sum'] = flower_size
    elif i == k-1 and 1 < n <= k+1: # Johnson graph special case
        data[ki][nn]['min_sum'] = k+1
    elif n <= 1 + k//(k-i): # Star special case
        # print("This is an automatic antiflower")
        data[ki][nn]['min_sum'] = 2*k-i
    else: # General case
        V=np.array([[int(x) for x in format(j, '0{}b'.format(n))] for j in range(1,2**n)]).T

        # Regions contained in each pair of vertices should sum to q
        Q=[]
        for j1 in range(n):
            for j2 in range(j1):
                Q.append(V[j1]&V[j2])
        Q=np.array(Q)
        
        if n == 1:
            Q = np.zeros((0,1),dtype=int)

        # Symmetry breaking constraint
        SB = np.zeros((n-1,2**n-1),dtype=int)
        for j in range(n-1):
            SB[j] = V[j+1]-V[j]
        SB = SB*np.power(4,np.sum(V,axis=0,keepdims=True))

        A = np.vstack([V,Q])

        # Reorder the columns of A, V, Q, and SB
        r = np.sum(V, axis=0)
        idx=np.argsort(r,kind='stable')
        splits = [math.comb(n, j) for j in range(1,n+1)]
        splits = np.cumsum(splits)
        splits = [0] + splits.tolist()
        A = A[:,idx]
        V = V[:,idx]
        Q = Q[:,idx]
        SB = SB[:,idx]

        # Right-hand side of equality constraints
        b = k*np.ones(n*(n+1)//2, dtype=int)
        b[n:] = i

        # Find the minimum sum
        min_sum = solve_min_sum(A, b, C=SB)
        data[ki][nn]['min_sum'] = min_sum

    print(f"Minimum sum: {data[ki][nn]['min_sum']}")

    # Check if this is a flower 
    min_sum = data[ki][nn]['min_sum']
    if min_sum == None: 
        pass
    elif min_sum == flower_size:
        data[ki][nn]['flower'] = True
        data[ki][nn]['type'] = 'flower'
        print("This is a flower.")
    elif min_sum == 2*k-i:
        data[ki][nn]['type'] = 'antiflower'
        print("This is an antiflower.")
    else:
        data[ki][nn]['flower'] = False
        data[ki][nn]['type'] = 'misc'
        data[ki]['omega star lb'] = max(data[ki]['omega star lb'],n)

    with open("clique_data.json", "w") as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ': '), indent=4)

    ###################
    # Check conjectures
    ###################
    q = (n-1)*i/k
    continuous_min = (n*k/math.ceil(q+1))*(1+(math.ceil(q)-q)/math.floor(q+1))
    data[ki][n]['continuous_min'] = continuous_min

    if continuous_min - math.floor(continuous_min) < 1e-10:
        expected_min = math.floor(continuous_min)
    else:
        expected_min = math.ceil(continuous_min)

    if min_sum != expected_min:
        data[ki][nn]['disaster'] = True
    else:
        data[ki][nn]['disaster'] = False

    # Save data to json
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
    # elif data[ki][nn]['flower']:
    elif data[ki][nn]['type'] == 'flower':
        solutions = (k-i)*np.ones((1,2**n-1),dtype=int)
        solutions[0,-1] = i
        np.savez_compressed(f"new_cliques/{ki}_{n}.npy", np.zeros((0,n),dtype=int))
        data[ki][nn]['all solutions found'] = True
    else:
        solutions = find_all_solutions_with_min_sum(A, b, min_sum, C=SB, time_limit=time_limit, old_solutions=[])
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