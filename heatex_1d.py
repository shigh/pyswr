
import argparse
import numpy as np
from mpi4py import MPI
from region import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--steps", help="Number of iterations",
                    type=int, default=1)
parser.add_argument("-x", "--nx", help="Total number of x points (global)",
                    type=int, default=100)
parser.add_argument("-t", "--nt", help="Total number of t points (global)",
                    type=int, default=100)
parser.add_argument("-o", "--overlap", help="overlap",
                    type=int, default=2)
parser.add_argument("--plot", help="Plot Results",
                    action="store_true")
parser.add_argument("--error", help="Print the error",
                    action="store_true")
args = parser.parse_args()


comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

overlap = args.overlap
offset  = overlap*2
nt = args.nt
nx = args.nx

dt = 1./(nt-1)
dx = 2.*np.pi/(nx-1)

x_vals_full = np.linspace(0, 2*np.pi, nx)
x_vals      = region_views(x_vals_full, size, overlap)[rank]
f0          = np.sin(x_vals)

A = one_d_heat_btcs(len(f0), dx, dt)
solver = ImplicitSolver1D(A)
r      = Region(nt, len(f0))
solver.x[:]    = f0
r.slices[0][:] = f0

has_left  = rank>0
has_right = rank<size-1
 
right = rank+1
left  = rank-1
    
if has_right:
    r.add_col(0, -1-offset)
if has_left:
    r.add_col(0, offset)
    
for _ in range(args.steps):
    solver.x[:] = f0
    
    for i in range(1, nt):
        solver.left  = r.cols[0][0][i]
        solver.right = r.cols[0][-1][i]
        solver.solve()
        r.update_cols(i, solver.x)
        
    send_requests = []
    if has_right:
        rr = comm.Isend(r.cols[0][-1-offset], dest=right)
        send_requests.append(rr)
    if has_left:
        rl = comm.Isend(r.cols[0][offset], dest=left)
        send_requests.append(rl)
        
    if has_right:
        comm.Recv(r.cols[0][-1], source=right)
    if has_left:
        comm.Recv(r.cols[0][0], source=left)
        
    MPI.Request.Waitall(send_requests)


if args.error:    
    expected = np.exp(-1.)*f0    
    error = np.max(np.abs(r.slices[nt-1] - expected))
    all_error = comm.gather(error, root=0)

    if rank==0:
        print "Avg: %f Min: %f Max: %f" % (np.mean(all_error),
                                           np.min(all_error),
                                           np.max(all_error))

if args.plot:

    all_x_vals = comm.gather(x_vals, root=0)
    all_f_vals = comm.gather(r.slices[nt-1], root=0)

    if rank==0:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        for i, j in zip(all_x_vals, all_f_vals):
            ax.plot(i, j)

        plt.show()
