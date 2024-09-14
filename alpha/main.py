#from mpi4py import MPI

import argparse
import numpy as np 
import torch
from torch import nn
import sys
sys.path.append('../chi/')
from GraphFun import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import math
from collections import Counter
matplotlib.rcParams['animation.convert_path'] = 'magick'

parser = argparse.ArgumentParser()
parser.add_argument("--n-steps", type=int, default=800, help="Number of steps")
parser.add_argument("--T", type=float, default=0.02, help="Temperature")
parser.add_argument("--lam", type=float, default=0.2, help="Regularization parameter")
parser.add_argument("--representation", type=str, default="softmax", help="Representation")
parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
parser.add_argument("--batch-size", type=int, default=10, help="Batch size")
parser.add_argument("--make-movie", action="store_true", help="Make movie")
parser.add_argument("--vki", nargs='+', type=int, default=[5, 2, 0], help="v, k, i")
args = parser.parse_args()



n_steps = args.n_steps
T = args.T
lam = args.lam
representation = args.representation
lr = args.lr
B=args.batch_size
make_movie = args.make_movie
v,k,i = args.vki
print("=================")
print("J(",v,",",k,",",i,")")
print("=================\n")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ",device,"\n")

G = genJohnsonGraph(v,k,i)
A = getAdjArray(G)
A = torch.tensor(A,dtype=torch.float32, device=device)
N = A.shape[0] # Number of vertices

q=1.0
c = 28


mask = torch.zeros(N,N,device=device) # Initialize with zeros, no learning happens

# Init x
if representation == "softmax":
    x = torch.randn((B,N,1),requires_grad=True, device=device) # Logits
    # x.data = 10*x.data
elif representation == "quantum":
    x = torch.randn((B,N,1),requires_grad=True, device=device) # Logits
    # x.data = torch.nn.functional.normalize(x.data,dim=-1,p=2)
elif representation == "literal":
    x = torch.rand((B,N,1),requires_grad=True, device=device)
    for i in range(5):
        x.data = torch.clip(x.data,0,1)
        x.data = c*torch.nn.functional.normalize(x.data,dim=1,p=1)
elif representation == "experimental":
    x = torch.rand((B,N,1),requires_grad=True, device=device)
    # x.data = torch.nn.functional.normalize(x.data,dim=-1,p=q)
    # x.data = torch.abs(x.data)
    
# Optimizer
optimizer = torch.optim.Adam([x], lr=lr)
# optimizer = torch.optim.SGD([x], lr=lr)

# Minimize loss
vals = []
points =[]
losses = []
alpha = torch.linspace(1.0,0.0,n_steps+1,device=device)
for step in range(n_steps+1):
    optimizer.zero_grad()

    # Get probability (make values positive)
    if representation == "softmax":
        y = torch.sigmoid(x)
    elif representation == "quantum":
        y = x**2
    elif representation == "literal":
        # y = torch.abs(x)
        y = x
    elif representation == "experimental":
        y = torch.pow(torch.abs(x),q)

    p = c*y/torch.sum(y,dim=1,keepdim=True) # We don't want to normalize, since this would just make everything move to 1...
    # p=y

    if make_movie:
        points.append(p.detach()) # Collect points

    # This is like doing a 1-coloring, but we also try to make the overall sum of vertex values as high as possible...
        
    L1 = 0.5*torch.sum(A*(p@p.transpose(-1,-2)),dim=(-1,-2))   
    L2 = (p-0.5).sum((-1,-2))**2  
    L3 = torch.sum(p*(1-p),dim=(-1,-2))
    L4 = -torch.sum(p,dim=(-1,-2))
    L5 = 0.5*torch.sum(A*(p**2@p.transpose(-1,-2)**2),dim=(-1,-2))
    L6 = (c-torch.sum(p,dim=1))**2
    # multi_loss = L1 + lam*L2
    # multi_loss = L1 + lam*L4

    # multi_loss = L5 + 5.0*L4 #+ alpha[step]*L3
    # print(x.data)
    # assert(0)
    multi_loss = L1

    # multi_loss = 0.5*torch.sum((A-lam*torch.ones_like(A))*(p@p.transpose(-1,-2)),dim=(-1,-2)) # Covariance matrix

    # Masking
    if step%50==0:

        # Run test
        # _, max_indices = torch.max(p, dim=-1)
        # colorings = torch.nn.functional.one_hot(max_indices, num_classes=c).float()
        independent_set = (p>0.5).float()
        S=independent_set@independent_set.transpose(1,2)
        edge_violations = 0.5*torch.sum(A*S,dim=(-1,-2))
        true_best, _ =torch.min(edge_violations,0)

        best = torch.min(multi_loss)
        avg = torch.mean(multi_loss)

        true_avg = torch.mean(edge_violations)
        true_best = torch.min(edge_violations)

        print(f"Step {step}: Average loss: {true_avg.item():.2f} Min loss: {true_best.item():.2f}\n")

        if step<0.9*n_steps:

            mask = 1-(1-step/n_steps)*torch.rand_like(A)
     
            # lam*=0.9

            print("Still masking")
        else:
            mask = torch.ones_like(A)
            # lam = 0.0 # This makes sure we end up with an independent set
        torch.cuda.empty_cache()
    
    if make_movie:
        losses.append(multi_loss.detach())
        
    loss = torch.sum(multi_loss) 
    loss.backward()

    optimizer.step()

    # Renormalize
    # x.data = torch.clip(x.data,0,1)
    for i in range(5):
        x.data = torch.clip(x.data,0,1)
        x.data = c*torch.nn.functional.normalize(x.data,dim=1,p=1)

    # if representation == "quantum":
    #     x.data = torch.nn.functional.normalize(x.data,dim=-1,p=2)
    # elif representation == "literal":
    #     x.data = torch.nn.functional.normalize(x.data,dim=-1,p=1)
    #     # x.data = torch.abs(x.data)
    # elif representation == "experimental":
    #     x.data = torch.nn.functional.normalize(x.data,dim=-1,p=q)
    #     x.data = torch.abs(x.data) # Makes smoother
    

##################################
# _, max_indices = torch.max(p, dim=-1)
# colorings = torch.nn.functional.one_hot(max_indices, num_classes=c).float()

# minmax=colorings.max(-1)[0].min()
# if minmax.item()<0.999:
#     print("minmax",minmax.item())
#     print("Warning: Not a valid coloring")

independent_set = (p>0.5).float()
# print(independent_set.shape)
# assert(0)

edge_violations = 0.5*torch.sum(A*(independent_set@independent_set.transpose(1,2)),dim=(-1,-2))
best,idx=torch.min(edge_violations,0)
sorted, _ = edge_violations.sort()
print("Edge violations: ",sorted[:1000])
print("Best: ",best.item())
# plt.hist(edge_violations.cpu().numpy(),[0]+0.5*np.arange(1,args.batch_size+1))
counts = Counter(sorted.to(torch.int32).cpu().numpy())
plt.bar(counts.keys(),counts.values())
plt.savefig("hist.png")

if best<10.1:
    np.save('J('+str(v)+','+str(k)+','+str(i)+')_best_'+str(int(best.item()))+'.npy',p[idx].detach().cpu().numpy())

# Get class sizes
best_ind_set = independent_set[idx]
ind_set_size = torch.sum(best_ind_set).item()
print("Independent set size: ",ind_set_size)
print("Independent set proportion: ",ind_set_size/N)

if make_movie:
    points = torch.stack([point[idx] for point in points])
    # # print(points[-1].cpu().numpy())

    # # Get colorings
    # _, max_indices = torch.max(points, dim=-1)
    # coloring = torch.nn.functional.one_hot(max_indices, num_classes=c).float()
    independent_set = (p>0.5).float()

    losses = torch.round(0.5*torch.sum(A*(independent_set@independent_set.transpose(-1,-2)),dim=(-1,-2))).int()
    # losses = 0.5*torch.sum(A*(points@points.transpose(-1,-2)),dim=(-1,-2))
    losses = losses.cpu().numpy()
    points = points.cpu().numpy() # Collect points from best trial

    # Create random matrix the projects to 2 dimensions from N dimensions
    if c>3:
        R = np.random.randn(2,c)
    else:
        R = np.array([
            [1/np.sqrt(2),-1/np.sqrt(2),0],
            [1/np.sqrt(6),1/np.sqrt(6),-2/np.sqrt(6)]])

    in_plane = points@R.T 
    xmin,ymin=1.1*np.min(in_plane,axis=(0,1))
    xmax,ymax=1.1*np.max(in_plane,axis=(0,1))

    fig, ax = plt.subplots(figsize=(10,10))
    scat = ax.scatter(in_plane[0,:,0], in_plane[0,:,1],s=30)
    

    # Create a Line2D object for plotting the edges
    line = Line2D([], [], color='k', alpha=0.1, linewidth=1)
    ax.add_line(line)

    # Add edges between points i and j whenever A[i,j]==1
    edges = [(i,j) for i in range(N) for j in range(i) if A[i, j] == 1]

    ax.set_title(f"Loss: {losses[0]: >10} \nFinal loss: {best.item()}", loc='left')
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])

    def update(frame_number):
        scat.set_offsets(in_plane[frame_number])

        x_pairs = [in_plane[frame_number,edge,0] for edge in edges]
        y_pairs = [in_plane[frame_number,edge,1] for edge in edges]
        line.set_data(x_pairs, y_pairs)

        ax.set_title(f"Step: {frame_number} \nLoss: {losses[frame_number]: >10} \nFinal loss: {best.item()}", loc='left')

    ani = animation.FuncAnimation(fig, update, frames=len(points), interval=50)
    ani.save('test.gif', writer='pillow', fps=10) # fps = 30
    plt.close(fig)
