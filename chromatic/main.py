#from mpi4py import MPI

import argparse
import numpy as np 
import torch
from torch import nn
import sys
sys.path.append('..')
from GraphFun import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
matplotlib.rcParams['animation.convert_path'] = 'magick'

parser = argparse.ArgumentParser()
parser.add_argument("--n-steps", type=int, default=800, help="Number of steps")
parser.add_argument("--c", type=int, default=9, help="Number of colors")
parser.add_argument("--beta", type=float, default=0.02, help="Temperature")
parser.add_argument("--T", type=float, default=0.02, help="Temperature")
parser.add_argument("--lam", type=float, default=1.0, help="Regularization parameter")
parser.add_argument("--representation", type=str, default="softmax", help="Representation")
parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
parser.add_argument("--batch-size", type=int, default=10, help="Batch size")
parser.add_argument("--make-movie", action="store_true", help="Make movie")
args = parser.parse_args()



n_steps = args.n_steps
c = args.c
# beta = args.beta
T = args.T
lam = args.lam
representation = args.representation
lr = args.lr
B=args.batch_size
make_movie = args.make_movie

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ",device)

G = genJohnsonGraph(10,4,3)
A = getAdjArray(G)
A = torch.tensor(A,dtype=torch.float32, device=device)
N = A.shape[0] # Number of vertices

mask = torch.ones(N,N,device=device)

# Init x
if representation == "softmax":
    x = torch.randn((B,N,c),requires_grad=True, device=device) # Logits
    x.data = 10*x.data
elif representation == "quantum":
    x = torch.randn((B,N,c),requires_grad=True, device=device) # Logits
    # x.data = 10*x.data
    x.data = torch.nn.functional.normalize(x.data,dim=-1)

elif representation == "experimental":
    x = torch.randn((B,N,c),requires_grad=True, device=device) # Logits
    x.data = 10*x.data
    x.data = torch.nn.functional.normalize(x.data,dim=-1)

    
# Optimizer
optimizer = torch.optim.Adam([x], lr=lr)
# optimizer = torch.optim.SGD([x], lr=lr)


# Minimize loss
vals = []
points =[]
losses = []

## Stuff I'm working on...
# # def edge_penalty(p):
# #     return 0
# def f(x):
#     # return (2*torch.asin(x)/np.pi)**2
#     return x**2
#     # return 1-torch.sqrt(1-x**2)
#     # return x**2/(1-x**2)

# Ts = torch.linspace(T,0.02,n_steps)
# Ts = torch.logspace(2,-1,n_steps)
for step in range(n_steps):
# while T>0.01:
    optimizer.zero_grad()

    # Get probability
    if representation == "softmax":
        beta = 1/T
        y = torch.exp(beta*x)

    elif representation == "quantum":
        beta = 1/T
        y = x**2
    # elif representation == "absolute":
    #     beta = 1/T
    #     y = torch.abs(x)

    if representation == "experimental":
        q = y**2
        p = q/torch.sum(q,dim=-1,keepdim=True)
    else:
        p = y/torch.sum(y,dim=-1,keepdim=True)
    # print(torch.min(y@y.transpose(-1,-2)))
    # print(torch.max(y@y.transpose(-1,-2)))

    if make_movie:
        points.append(p.detach()) # Collect points


    # Calculate loss
    multi_loss = 0.5*torch.sum((mask*A)*(p@p.transpose(-1,-2)),dim=(-1,-2)) # Covariance matrix
    # multi_loss = torch.sum(A*((p@p.transpose(-1,-2)))**3,dim=(-1,-2)) # Covariance matrix
    # multi_loss = torch.sum(A*(torch.asin(p@p.transpose(-1,-2))),dim=(-1,-2)) # Covariance matrix # Unstable

        # z = p #torch.abs(x)
        # w = torch.abs(x)
        # z = w/torch.sqrt(torch.sum(w**2,dim=-1,keepdim=True))
        # multi_loss = torch.sum((A*(z@z.transpose(-1,-2)))**2,dim=(-1,-2)) # Covariance matrix
        # multi_loss = torch.sum((A*(torch.asin(z@z.transpose(-1,-2))*2/np.pi)**2),dim=(-1,-2)) # Covariance matrix
        # multi_loss = torch.sum((A*f(z@z.transpose(-1,-2))),dim=(-1,-2)) # Covariance matrix

    if step%100==0:
        best = torch.min(multi_loss)
        print(f"Step {step}: Loss: {multi_loss.mean().item():.2f} Best: {best.item():.2f}")
        if step<0.9*n_steps:
            # mask = (torch.rand_like(A) < 0.9).float()
            # mask = 1-0.2*torch.rand_like(A)
            # mask = 1-0.4*torch.rand_like(A)
            mask = 1-(1-step/n_steps)*torch.rand_like(A)
        else:
            mask = torch.ones_like(A)
        torch.cuda.empty_cache()
    
    if make_movie:
        losses.append(multi_loss.detach())
        
    loss = torch.sum(multi_loss) 
    loss.backward()
    # torch.nn.utils.clip_grad_norm_([x], max_norm=0.1,norm_type='inf')
    optimizer.step()
    if representation == "quantum" or "experimental":
        x.data = torch.nn.functional.normalize(x.data,dim=-1)
    

# print("Loss: ",loss.item())
# print("Independence: ",torch.sum(A*(p@p.t())).item())
# print("")
    

# Two versions
###############################
# probs = torch.softmax(x,dim=1)
# # Choose an element from each row
# indices = torch.multinomial(probs, num_samples=1)
# # Create a new tensor with one-hot encoding for each row
# coloring = torch.zeros_like(probs)
# coloring.scatter_(1, indices, 1)

# if representation == "softmax":
#     coloring = torch.softmax(20*x,dim=1)
# elif representation == "quantum":
#     # coloring = x**2
#     y = torch.exp(beta*x)
#     coloring = y/torch.sum(y,dim=1,keepdim=True)

coloring = torch.softmax(100*torch.abs(p),dim=-1)
##################################

minmax=coloring.max(1)[0].min(0)[0]

print(minmax)
# coloring = torch.softmax(beta*x,dim=1)
# print(p)
# print(p>0.5)
val = torch.sum(A*(coloring@coloring.transpose(1,2)),dim=(-1,-2))
best,idx=torch.min(val,0)
# print("Final losses: ",val)
print("Best: ",best.item())
if best<10.1:
    np.save('best_'+str(int(best.item()))+'.npy',p>0.5)

if make_movie:
    points = torch.stack([point[idx] for point in points])

    # Get 
    coloring = torch.softmax(50*points,dim=-1)  # Get actual colorings
    points = points.cpu().numpy() # Collect points from best trial
    losses = torch.round(0.5*torch.sum(A*(coloring@coloring.transpose(-1,-2)),dim=(-1,-2))).int()
    losses = losses.cpu().numpy()

    # losses = torch.stack([loss[idx] for loss in losses]).cpu().numpy() # Collect losses from best trial


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

    fig, ax = plt.subplots()
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
    ani.save('test3.gif', writer='imagemagick', fps=30)
    plt.close(fig)
