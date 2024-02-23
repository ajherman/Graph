import torch
import numpy as np

# We will have energy represent the number of edge violations for a coloring

# Petersen graph adjacency matrix
A = torch.tensor([
    [0,1,0,0,1,1,0,0,0,0],
    [1,0,1,0,0,0,1,0,0,0],
    [0,1,0,1,0,0,0,1,0,0],
    [0,0,1,0,1,0,0,0,1,0],
    [1,0,0,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,1,1,0],
    [0,1,0,0,0,0,0,0,1,1],
    [0,0,1,0,0,1,0,0,0,1],
    [0,0,0,1,0,1,1,0,0,0],
    [0,0,0,0,1,0,1,1,0,0]
],dtype=torch.float32)

A = torch.tensor(A,dtype=torch.float32)

# Iterate over all binary 10-tuples
for i in range(2**10):
    # Convert to binary
    x = torch.tensor([int(j) for j in bin(i)[2:].zfill(10)],dtype=torch.float32)
    # Calculate energy
    print(x)
    if i>10:
        assert(0)