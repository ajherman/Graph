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
idx=np.argsort(r)

splits = [math.comb(n, i) for i in range(1,n+1)]
splits = np.cumsum(splits)
splits = [0] + splits.tolist()

A = A[:,idx]

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

def find_all_solutions_with_min_sum(A, b, min_sum):
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
        def __init__(self, variables):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self._variables = variables
            self._solutions = []

        def on_solution_callback(self):
            solution = [self.Value(var) for var in self._variables]
            self._solutions.append(solution)

        def get_solutions(self):
            return np.array(self._solutions)

    # Create the solution collector
    collector = SolutionCollector(x)

    # Search for all solutions
    solver.SearchForAllSolutions(model, collector)

    # Return the list of all solutions
    return collector.get_solutions()


# Step 1: Find the minimum sum
min_sum = solve_min_sum(A, b)

print("\nMinimum sum:", min_sum)

# Step 2: Find all solutions with the minimum sum
if min_sum is not None:
    solutions = find_all_solutions_with_min_sum(A, b, min_sum)
    print(f"\nAll solutions with sum {min_sum}:")
    # sol_sum = np.sum(solutions, axis=0)
    # idx = np.where(sol_sum>0)
    # solutions = solutions[:,idx]
    with open(f"cliques/{n}-cliques.txt", "a") as f:
        f.write(f"\nk={k},i={i} : \n")
        profile_str = ['' for j in range(len(solutions))]
        for i in range(len(splits)-1):
            region = solutions[:,splits[i]:splits[i+1]]
            if np.sum(region)>0:
                print(f"\n{i+1}-intersecting regions:")
                for j in range(len(region)):
                    profile_str[j] += "| <"+str(i+1)+"> vals:"+str(np.unique(region[j]))[1:-1]+", total:"+str(np.sum(region[j]))+" |"
                print(region)  
                # Write to log file
                # f.write(f"{i+1} ") # This was original
            else:
                pass
        for x in set(profile_str):
            # f.write(str(i))
            f.write(x)
            f.write("\n")
        f.write("\n")
        f.close()
            # print(f"\n{i+1}-intersecting regions:")
            # print("None")
    # for sol in solutions:
    #     print(sol)
else:
    print("No solution found.")

toc = time.time()
print("Elapsed time:", toc-tic, "seconds.")