import numpy as np
from itertools import combinations_with_replacement as cwr
from ortools.sat.python import cp_model
import math
import time
import argparse

np.set_printoptions(threshold=np.inf)

# from mpi4py import MPI

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Enumerate all cliques in a graph.")
parser.add_argument("--n", type=int, default=6, help="Clique size")
parser.add_argument("--k", type=int, default=8, help="Vertex size")
parser.add_argument("--i", type=int, default=3, help="Intersection size")

args = parser.parse_args()

n,k,i = args.n, args.k, args.i

time_limit = 60

print(f"n = {n}, k = {k}, i = {i}")

tic = time.time()

v = (k-i)*n+i # Flower
# Iterate over all 2^n-1 tuples with non-negative integer values summing to at most v

# Bit representations of each region (also used to enforce that each vertex region should sum to 1)
V=np.array([[int(x) for x in format(j, '0{}b'.format(n))] for j in range(1,2**n)]).T

# Regions contained in each pair of vertices should sum to q
Q=[]
for j1 in range(n):
    for j2 in range(j1):
        Q.append(V[j1]&V[j2])
Q=np.array(Q)

A = np.vstack([V,Q])

r = np.sum(V, axis=0)
idx=np.argsort(r,kind='stable')

splits = [math.comb(n, i) for i in range(1,n+1)]
splits = np.cumsum(splits)
splits = [0] + splits.tolist()

A = A[:,idx]
V = V[:,idx]
Q = Q[:,idx]

# print(V[:,idx])

# Right-hand side of equality constraints
b = k*np.ones(n*(n+1)//2, dtype=int)
b[n:] = i

def solve_min_sum(A, b):
    # Initialize the constraint programming model
    model = cp_model.CpModel()

    num_rows = len(A)
    num_vars = len(A[0])
    
    # Create non-negative integer variables
    x = [model.NewIntVar(0, max(b[-1],b[0]-b[-1]), f'x[{j}]') for j in range(num_vars)]

    # Add constraints Ax = b
    for i in range(num_rows):
        model.Add(sum(A[i][j] * x[j] for j in range(num_vars)) == b[i])

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

def find_all_solutions_with_min_sum(A, b, min_sum, time_limit=1e10):
    # Initialize the constraint programming model
    model = cp_model.CpModel()

    num_rows = len(A)
    num_vars = len(A[0])

    # Create non-negative integer variables
    x = [model.NewIntVar(0, max(b[-1],b[0]-b[-1]), f'x[{i}]') for i in range(num_vars)]

    # Add constraints Ax = b
    for i in range(num_rows):
        model.Add(sum(A[i][j] * x[j] for j in range(num_vars)) == b[i])

    # Add constraint that the sum of x equals the minimum sum
    model.Add(sum(x) == min_sum)

    # Create the solver
    solver = cp_model.CpSolver()

    # Collect all solutions
    solutions = []

    # Define a solution callback to collect solutions
    class SolutionCollector(cp_model.CpSolverSolutionCallback):
        def __init__(self, variables, time_limit):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self._variables = variables
            self._solutions = []
            self.creation_time = time.time()
            self.time_limit = time_limit

        def on_solution_callback(self):
            solution = [self.Value(var) for var in self._variables]
            self._solutions.append(solution)
            if time.time()-self.creation_time>self.time_limit:
                self.StopSearch()

        def get_solutions(self):
            return np.array(self._solutions)

    # Create the solution collector
    collector = SolutionCollector(x,time_limit=time_limit)

    # Search for all solutions
    solver.SearchForAllSolutions(model, collector)

    # Return the list of all solutions
    return collector.get_solutions()


# Step 1: Find the minimum sum
time0 = time.time()
min_sum = solve_min_sum(A, b)

print("\nMinimum sum:", min_sum)



# Step 2: Find all solutions with the minimum sum
if min_sum is not None:
    time1 = time.time()

    with open(f"cliques/{n}-cliques.txt", "a") as f:
        f.write(f"\nn={n},k={k},i={i} : min v = {min_sum} \n")

        # Check conjecture
        q = (n-1)*i/k
        expected_min = (n*k/math.ceil(q+1))*(1+(math.ceil(q)-q)/math.floor(q+1))
        before_rounding = expected_min
        if expected_min - math.floor(expected_min) < 1e-10:
            expected_min = math.floor(expected_min)
        else:
            expected_min = math.ceil(expected_min)
        if min_sum != expected_min:
            f.write(f"Disaster: n={n}, k={k}, i={i}, before rounding={before_rounding}\n")
            f.write(f"Expected min: {expected_min}\n")
            f.write(f"Actual min: {min_sum}\n")
        time2 = time.time()
        # Find and analyze all solutions
        solutions = find_all_solutions_with_min_sum(A, b, min_sum, time_limit=time_limit)
        all_ob = True
        all_ce = True
        print(f"\nAll solutions with sum {min_sum}:")
        time3 = time.time()
        profiles = []
        for solution in solutions:
            counter_example = False
            oddball = False

            profile_str = ''
            for i in range(len(splits)-1):
                region = solution[splits[i]:splits[i+1]]
                unique_vals = np.unique(region)
                if 0 in unique_vals and len(unique_vals)==1:
                    continue
                else:
                    if not np.floor(q)<=i<=np.ceil(q):
                        counter_example = True

                    vertex_sizes=np.unique(V[:,splits[i]:splits[i+1]]@region)
                    inter_sizes = np.unique(Q[:,splits[i]:splits[i+1]]@region)

                    if len(vertex_sizes)>1:
                        oddball = True

                    profile_str += f"| <{i+1}> vals:{unique_vals}, total:{np.sum(region)}, v sizes:{vertex_sizes}, i sizes:{inter_sizes} |"
            if counter_example:
                profile_str += " COUNTER-EXAMPLE"
            else:
                all_ce = False
            if oddball:
                profile_str += " ODD-BALL"
            else:
                all_ob = False
            profile_str += "\n"
            profiles.append(profile_str)
        for profile_str in set(profiles):
            f.write(profile_str)
        if all_ob:
            f.write("All odd-balls\n")
        if all_ce:
            f.write("All counter-examples\n")
        f.write("\n")
        f.close()
    time4 = time.time()
    for i in range(len(splits)-1):
        region = solutions[:,splits[i]:splits[i+1]]
        if np.sum(region)>0:
            if not np.floor(q)<=i+1<=np.ceil(q):
                pass
            print(f"\n{i+1}-intersecting regions:")
            print(region)  
    time4 = time.time()
    # print("Time to find min sum:", time1-time0)
    # print("Time to find all solutions:", time2-time1)
    # print("Time to look for features:", time3-time2)
    # print("Time to print solutions:", time4-time3)
        
else:
    print("No solution found.")



# # Step 2: Find all solutions with the minimum sum
# if min_sum is not None:
#     solutions = find_all_solutions_with_min_sum(A, b, min_sum)
#     print(f"\nAll solutions with sum {min_sum}:")
#     # sol_sum = np.sum(solutions, axis=0)
#     # idx = np.where(sol_sum>0)
#     # solutions = solutions[:,idx]

#     with open(f"cliques/{n}-cliques.txt", "a") as f:
#         f.write(f"\nk={k},i={i} : \n")
#         q = (n-1)*i/k
#         profile_str = ['' for j in range(len(solutions))] # Profile of cliques
#         counter_example = [False for j in range(len(solutions))] # Whether a clique is a counter-example
#         for i in range(len(splits)-1):
#             region = solutions[:,splits[i]:splits[i+1]]
#             if np.sum(region)>0:
#                 if not np.floor(q)<=i+1<=np.ceil(q):
#                     pass
#                 print(f"\n{i+1}-intersecting regions:")
#                 for j in range(len(region)):
#                     profile_str[j] += "| <"+str(i+1)+"> vals:"+str(np.unique(region[j]))[1:-1]+", total:"+str(np.sum(region[j]))+" |"
#                 print(region)  
#                 # Write to log file
#                 # f.write(f"{i+1} ") # This was original
           
#         for x in set(profile_str):
#             # f.write(str(i))
#             f.write(x)
#             f.write("\n")
#         f.write("\n")
#         f.close()
#             # print(f"\n{i+1}-intersecting regions:")
#             # print("None")
#     # for sol in solutions:
#     #     print(sol)
# else:
#     print("No solution found.")




toc = time.time()
print("Elapsed time:", toc-tic, "seconds.")