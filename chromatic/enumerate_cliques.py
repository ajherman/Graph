import numpy as np
from itertools import combinations_with_replacement as cwr
import math

n=6
k,i = 12,4

# x = [0]*(2**n-1)
# for j in range(i):
#     x[0] = j
#     for c in cwr(range(3),5):
#         x[1:n+1] = c


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

A = np.vstack([V,Q])

r = np.sum(V, axis=0)
idx=np.argsort(r)
# print(r)
# print(len(r))
# print(idx)

splits = [math.comb(n, i) for i in range(1,n+1)]
splits = np.cumsum(splits)
splits = [0] + splits.tolist()
# print(splits)
# assert(0)

A = A[:,idx]
# assert(0)

# Right-hand side of equality constraints
b = k*np.ones(n*(n+1)//2, dtype=int)
b[n:] = i

from ortools.sat.python import cp_model

def solve_min_sum(A, b):
    # Initialize the constraint programming model
    model = cp_model.CpModel()

    num_rows = len(A)
    num_vars = len(A[0])
    
    # Create non-negative integer variables
    x = [model.NewIntVar(0, 12, f'x[{j}]') for j in range(num_vars)]

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
    x = [model.NewIntVar(0, max(b), f'x[{i}]') for i in range(num_vars)]

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

# Step 2: Find all solutions with the minimum sum
if min_sum is not None:
    solutions = find_all_solutions_with_min_sum(A, b, min_sum)
    print(f"All solutions with minimum sum {min_sum}:")
    # sol_sum = np.sum(solutions, axis=0)
    # idx = np.where(sol_sum>0)
    # solutions = solutions[:,idx]

    for i in range(len(splits)-1):
        print(f"{i+1}-intersecting regions:")
        region = solutions[:,splits[i]:splits[i+1]]
        if np.sum(region)>0:
            print(region)
        else:
            print("None")
    # for sol in solutions:
    #     print(sol)
else:
    print("No solution found.")


# print(A_eq)

# x = [0]*(2**n-1)

# idx=0
# b = [0]*(n*(n+1)//2)
# while True:
#     if 