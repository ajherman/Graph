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
import math
from collections import Counter

def main(args):
    matplotlib.rcParams['animation.convert_path'] = 'magick'

    # Memory analysis
    print("Initial GPU Usage")
    print("Allocated:", round(torch.cuda.memory_allocated(0)/1024**3,1), "GB")
    print("Reserved: ", round(torch.cuda.memory_reserved(0)/1024**3,1), "GB")

    n_steps = args.n_steps
    c = args.c
    T = args.T
    lam = args.lam
    representation = args.representation
    lr = args.lr
    B=args.batch_size
    make_movie = args.make_movie
    v,k,i = args.vki
    # maxcut = args.maxcut
    sparse = args.sparse
    if c > 2:
        maxcut = False

    print("=================")
    print("J(",v,",",k,",",i,")")
    print("=================\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ",device,"\n")

    G = genJohnsonGraph(v,k,i)
    A = getAdjArray(G)
    A = torch.tensor(A,dtype=torch.float32, device=device, requires_grad=False)
    if sparse:
        # # A = A.to_sparse()
        # coo = A.to_sparse()
        # indices = coo.indices()
        # values = coo.values()

        # row_indices = indices[0]
        # col_indices = indices[1]
        # A = torch.sparse_csr_tensor(row_indices, col_indices, values, A.size())

        # Get index arrays for adjacent vertex pairs
        idx1, idx2 = [],[]
        for v1 in range(A.shape[0]):
            for v2 in range(i):
                if A[v1,v2] != 0:
                    idx1.append(v1)
                    idx2.append(v2)
        idx1 = torch.tensor(idx1, device=device)
        idx2 = torch.tensor(idx2, device=device)
    
    A = A.to(device) # Necessar?
    N = A.shape[0] # Number of vertices

    q=1.0

    mask = torch.zeros(N,N,device=device) # Initialize with zeros, no learning happens

    # Init x
    if representation == "softmax":
        x = torch.randn((B,N,c),requires_grad=True, device=device) # Logits
        x.data = 10*x.data
    elif representation == "quantum":
        x = torch.randn((B,N,c),requires_grad=True, device=device) # Logits
        x.data = torch.nn.functional.normalize(x.data,dim=-1,p=2)
    elif representation == "literal":
        x = torch.rand((B,N,c),requires_grad=True, device=device)
        x.data = torch.nn.functional.normalize(x.data,dim=-1,p=1)
    elif representation == "experimental":
        x = torch.rand((B,N,c),requires_grad=True, device=device)
        x.data = torch.nn.functional.normalize(x.data,dim=-1,p=q)
        # x.data = torch.abs(x.data)
    elif representation == "maxcut":
        assert(c==2)
        x = torch.rand((B,N,1),requires_grad=True, device=device)
        x.data = x.data[:,:,:1]

    # Optimizer
    optimizer = torch.optim.Adam([x], lr=lr)
    # optimizer = torch.optim.SGD([x], lr=lr)

    # Minimize loss
    vals = []
    points =[]
    losses = []
    for step in range(n_steps+1):
        optimizer.zero_grad()

        # Get probability (make values positive)
        if representation == "softmax":
            y = torch.exp(x/T)
        elif representation == "quantum":
            y = x**2
        elif representation == "literal":
            y = torch.abs(x)
        elif representation == "experimental":
            y = torch.pow(torch.abs(x),q)
        elif representation == "maxcut":
            # y = torch.sigmoid(x)
            y = x

        # denom = torch.sum(y,dim=-1,keepdim=True)

        if representation == "maxcut":
            p = y
        else:
            p = y/torch.sum(y,dim=-1,keepdim=True) #torch.sum(y,dim=-1,keepdim=True) # If already normalized, this should not change anything (but might affect the gradient)
        

        if make_movie:
            points.append(p.detach()) # Collect points

        # Calculate loss (without masking, this should be equal to the number of edge violations...)
        # if representation == "softmax" or "quantum":
        #     multi_loss = 0.5*torch.sum((mask*A)*(p@p.transpose(-1,-2)),dim=(-1,-2)) # Covariance matrix
        # elif representation == "literal":

        if representation == "maxcut":
            
            # # Original
            # multi_loss = 0.5*torch.sum((mask*A)*(p@p.transpose(-1,-2)+(1-p)@(1-p.transpose(-1,-2))),dim=(-1,-2)) # Covariance matrix
            
            # Laplacian (new, experimental)
            L = torch.diag(torch.sum(A,dim=-1)) - A
            multi_loss = -0.5*torch.sum((mask*L)*(p@p.transpose(-1,-2)))
        else:
            if sparse:
                # A_indices = A._indices().t()
                # rows = A_indices[:, 0]
                # cols = A_indices[:, 1]
                # Q_values = (p[rows] * p[cols]).sum(dim=-1)
                # multi_loss = 0.5 * Q_values.sum()
                
                multi_loss = torch.sum(p[idx1]*p[idx2])
            else:
                multi_loss = 0.5*torch.sum((mask*A)*(p@p.transpose(-1,-2)),dim=(-1,-2)) # Covariance matrix
        

        # multi_loss_alt=0.5*torch.sum(mask*A*(p@p.transpose(-1,-2)),dim=(-1,-2))            
        # multi_loss = 0.5*torch.sum(torch.stack([(mask*A)@x for x in p])*p,dim=(-1,-2)) # Covariance matrix
        # multi_loss = 0.5*torch.sum(torch.stack([(mask*A)@x for x in p])*p,dim=(-1,-2)) # Covariance matrix
        # multi_loss = torch.sum(A*((p@p.transpose(s-1,-2))**3,dim=(-1,-2)) # Covariance matrix
        # multi_loss = torch.sum(A*(torch.asin(p@p.transpose(-1,-2))),dim=(-1,-2)) # Covariance matrix # Unstable
        # z = p #torch.abs(x)
        # w = torch.abs(x)
        # z = w/torch.sqrt(torch.sum(w**2,dim=-1,keepdim=True))
        # multi_loss = torch.sum((A*(z@z.transpose(-1,-2)))**2,dim=(-1,-2)) # Covariance matrix
        # multi_loss = torch.sum((A*(torch.asin(z@z.transpose(-1,-2))*2/np.pi)**2),dim=(-1,-2)) # Covariance matrix
        # multi_loss = torch.sum((A*f(z@z.transpose(-1,-2))),dim=(-1,-2)) # Covariance matrix

        # Masking
        if step%500==0:

            # Run test
            _, max_indices = torch.max(p, dim=-1)

            if representation == "maxcut":
                colorings = (p>0.5).float()
            else:
                colorings = torch.nn.functional.one_hot(max_indices, num_classes=c).float()

            # print(p.shape)
            # print(colorings.shape)
            # assert(0)

            edge_violations = 0.5*torch.sum(A*(colorings@colorings.transpose(-1,-2)+(1-colorings)@(1-colorings.transpose(-1,-2))),dim=(-1,-2))
            # if sparse:
            #     edge_violations = edge_violations.to_dense()
            true_best, _ =torch.min(edge_violations,0)

            best = torch.min(multi_loss)
            avg = torch.mean(multi_loss)

            print(f"Step {step}: Average loss: {avg.item():.2f} Min loss: {true_best.item():.2f}\n")

            if step<0.9*n_steps:
                # mask = (torch.rand_like(A) < 0.9).float()
                # mask = 1-0.2*torch.rand_like(A)
                # mask = 1-0.4*torch.rand_like(A)

                # mask = 1-(1-step/n_steps)*torch.rand_like(A)
                mask = 1-(1-step/n_steps)*torch.rand(A.shape)
                mask = mask.to(device)
                print("Still masking")
            else:
                mask = torch.ones_like(A)
            torch.cuda.empty_cache()
        
        if make_movie:
            losses.append(multi_loss.detach())
            
        loss = torch.sum(multi_loss) 
        loss.backward()

        optimizer.step()

        # Renormalize
        if representation == "quantum":
            x.data = torch.nn.functional.normalize(x.data,dim=-1,p=2)
        elif representation == "literal":
            x.data = torch.nn.functional.normalize(x.data,dim=-1,p=1)
            # x.data = torch.abs(x.data)
        elif representation == "experimental":
            x.data = torch.nn.functional.normalize(x.data,dim=-1,p=q)
            x.data = torch.abs(x.data) # Makes smoother
        elif representation == "maxcut":
            x.data = torch.clamp(x.data,0,1)
        

    ##################################
    _, max_indices = torch.max(p, dim=-1)
    colorings = torch.nn.functional.one_hot(max_indices, num_classes=c).float()
    minmax=colorings.max(-1)[0].min()
    if minmax.item()<0.999:
        print("minmax",minmax.item())
        print("Warning: Not a valid coloring")

    if representation == "maxcut":
        edge_violation = 0.5*torch.sum((A)*(p@p.transpose(-1,-2)+(1-p)@(1-p.transpose(-1,-2))),dim=(-1,-2)) # Covariance matrix
    else:
        edge_violations = 0.5*torch.sum(A*(colorings@colorings.transpose(1,2)),dim=(-1,-2))
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
    best_coloring = colorings[idx]
    # print(best_coloring)
    class_sizes = torch.sum(best_coloring,dim=0)
    print("Class sizes: ",class_sizes)
    print("Class proportions: ",class_sizes/N)

    # Poljak-Tuza conjecture bound
    if c==2: # Cobnjecture only applies to 2-colorings
        possible=[]
        for p in range(v-k):
            possible.append( math.comb(v-p,k)*(math.comb(v-k,k)-math.comb(v-k-p,k)) )
        pt_bound = 0.5*math.comb(v,k)*math.comb(k,i)*math.comb(v-k,k-i) - np.max(possible)
        print("Poljak Tuza: ",pt_bound)
        if best.item()<pt_bound:
            print("Counterexample to Poljak-Tuza conjecture: J(",v,",",k,",",i,")")
            if v<=4.3*k-0.3333333333333333:
                print("Counterexample to theorem")
            else:
                print("Not a counterexample to theorem")
            print("Best found: ",best.item())
            print("Poljak Tuza bound: ",pt_bound)
            np.save('counterexample_J('+str(v)+','+str(k)+','+str(i)+').npy',best_coloring.detach().cpu().numpy())

    # Memory analysis
    print("GPU Usage after running the code")
    print("Allocated:", round(torch.cuda.memory_allocated(0)/1024**3,1), "GB")
    print("Reserved: ", round(torch.cuda.memory_reserved(0)/1024**3,1), "GB")

    if make_movie:
        points = torch.stack([point[idx] for point in points])
        # print(points[-1].cpu().numpy())

        # Get colorings
        _, max_indices = torch.max(points, dim=-1)
        coloring = torch.nn.functional.one_hot(max_indices, num_classes=c).float()

        losses = torch.round(0.5*torch.sum(A*(coloring@coloring.transpose(-1,-2)),dim=(-1,-2))).int()
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=800, help="Number of steps")
    parser.add_argument("--c", type=int, default=9, help="Number of colors")
    parser.add_argument("--T", type=float, default=0.02, help="Temperature")
    parser.add_argument("--lam", type=float, default=1.0, help="Regularization parameter")
    parser.add_argument("--representation", type=str, default="softmax", help="Representation")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size")
    parser.add_argument("--make-movie", action="store_true", help="Make movie")
    parser.add_argument("--vki", nargs='+', type=int, default=[5, 2, 0], help="v, k, i")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--sparse", action="store_true", help="If sparse, use sparse A")
    # parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # bottleneck.main(args)
    main(args)