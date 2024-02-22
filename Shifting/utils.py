import torch
import numpy as np

F = torch.zeros(5,5,dtype=torch.bool).to(torch.bool)
for n,(i,j) in enumerate([(0,3),(1,3),(1,0),(2,4)]):
    F[n,[i,j]] = True

print(F)

def GetHash(F):
    N,v = F.shape
    powers = torch.pow(2,torch.arange(v)) # Powers of 2
    return F.to(torch.long)@powers

def Shift(F,i,j): # F is a Nxv array of Booleans
    idx_j_not_i = torch.where(F[:, j] & ~F[:, i])[0] # These are the sets that might move
    idx_i_not_j = torch.where(F[:, i] & ~F[:, j])[0]
    # print(F[torch.tensor([],dtype=torch.long)])
    
    try_to_move = F[idx_j_not_i].clone() # The subset we are trying to move
    try_to_move[:,[i,j]] = torch.BoolTensor([True, False])
    occupied = F[idx_i_not_j].clone() # The subset that is blocking the move

    occupied_hash = GetHash(occupied)
    try_to_move_hash = GetHash(try_to_move)

    idx2 = torch.where(~torch.isin(try_to_move_hash,occupied_hash)) # Get indices of sets that can move
    can_move = idx_j_not_i[idx2]
    # print(i,j)
    # print("move",can_move)
    # print(F[can_move][:,[i,j]].shape)
    # print(F[can_move])
    # F[can_move][:,[i,j]] = torch.BoolTensor([True, False])
    F[can_move,i] = True
    F[can_move,j] = False
    # print(F[can_move][:,[i,j]])
    # # print(try_to_move)
    # # idx_i_not_j = torch.where(F[:, i] & ~F[:, j])[0]
    # # attempt = F.clone() 
    # # attempt[idx_j_not_i][:,i], attempt[idx_j_not_i][:,j] = True, False
    # # print(attempt)
    # print(F)
    



for j in range(5):
    for i in range(j):
        print(i,j)
        Shift(F,i,j)
        print(F)
# print(F)

for row in F:
    print(torch.where(row)[0]+1)


