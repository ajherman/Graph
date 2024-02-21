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
matplotlib.rcParams['animation.convert_path'] = 'magick'

parser = argparse.ArgumentParser()
parser.add_argument("--n-steps", type=int, default=800, help="Number of steps")
parser.add_argument("--c", type=int, default=9, help="Number of colors")
parser.add_argument("--beta", type=float, default=0.02, help="Temperature")
parser.add_argument("--T", type=float, default=0.02, help="Temperature")
parser.add_argument("--lam", type=float, default=1.0, help="Regularization parameter")
parser.add_argument("--representation", type=str, default="softmax", help="Representation")
parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
args = parser.parse_args()

n_steps = args.n_steps
c = args.c
# beta = args.beta
T = args.T
lam = args.lam
representation = args.representation
lr = args.lr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class energyModel(torch.nn.Module):
#     def __init__(self, A, lam=1.0):
#         super(energyModel, self).__init__()
#         self.A = A
#         self.lam = lam

#     def forward(self, x):
#         energy = x.t()@self.A@x+self.lam*torch.norm(x,1)
#         return energy

G = genJohnsonGraph(10,4,3)
A = getAdjArray(G)

A = torch.tensor(A,dtype=torch.float32, device=device)
N = A.shape[0] # Number of vertices


# Init x
if representation == "softmax":
    x = torch.randn((N,c),requires_grad=True, device=device) # Logits
    x.data = 10*x.data
elif representation == "quantum":
    x = torch.randn((N,c),requires_grad=True, device=device) # Logits
    # x.data = 10*x.data
    x.data = torch.nn.functional.normalize(x.data,dim=1)
    
# Optimizer
optimizer = torch.optim.Adam([x], lr=lr)
# optimizer = torch.optim.SGD([x], lr=lr)


# Minimize loss
vals = []
points =[]
losses = []


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
        y = (beta*x)**2


    p = y/torch.sum(y,dim=1,keepdim=True)

    points.append(p.detach()) # Collect points

    # Calculate loss
    loss = torch.sum(A*(p@p.t())) # Covariance matrix
    # loss = torch.norm(A*(p@p.t())) # Covariance matrix

    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    if representation == "quantum":
        x.data = torch.nn.functional.normalize(x.data,dim=1)
    

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

coloring = torch.softmax(20*p,dim=1)
##################################

minmax=coloring.max(1)[0].min(0)[0].item()
print(minmax)
# coloring = torch.softmax(beta*x,dim=1)
# print(p)
print(p>0.5)
val = torch.sum(A*(coloring@coloring.t())).item()
print(val)
points = torch.stack(points).cpu().numpy()
# Create random matrix the projects to 2 dimensions from N dimensions
R = np.random.randn(2,c)

# R = np.array([[1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)],
#               [1/np.sqrt(2),-1/np.sqrt(2),0],
#               [1/np.sqrt(6),1/np.sqrt(6),-2/np.sqrt(6)]])

in_plane = points@R.T #-np.array([1]+[0]*(c-1))/np.sqrt(c)
xmin,ymin=1.1*np.min(in_plane,axis=(0,1))
xmax,ymax=1.1*np.max(in_plane,axis=(0,1))
# print(in_plane.shape)
# assert(0)
# Animate points of in_plane as moving scatter plot
# print(animation.writers.list())
# matplotlib.rcParams['animation.convert_path'] = '/usr/bin/convert'
# assert(0)

fig, ax = plt.subplots()
scat = ax.scatter(in_plane[0,:,0], in_plane[0,:,1],s=30)

ax.set_xlim([xmin,xmax])
ax.set_ylim([ymin,ymax])

def update(frame_number):
    scat.set_offsets(in_plane[frame_number])

ani = animation.FuncAnimation(fig, update, frames=n_steps, interval=50)
ani.save('test.gif', writer='imagemagick', fps=30)
plt.close(fig)
