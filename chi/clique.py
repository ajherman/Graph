from GraphFun import *
import torch
v,k= 20,10
G = genJohnsonGraph(v,k,k-1) #(10,4,3)
A = getAdjArray(G)
A = torch.tensor(A,dtype=torch.float32)
N = A.shape[0] # Number of vertices
M = torch.eye(N)-(A/(k*(v-k)))

x = torch.randn((N,1)) # Logits
for i in range(10000):
    y = M@x
    if i%1 == 0:
        print((1-(y/x))*k*(v-k))
    x=y
    x = x/torch.norm(x)
    # print(x)