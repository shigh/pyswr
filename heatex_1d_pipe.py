
import argparse
import numpy as np
from mpi4py import MPI
from region import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--steps", help="Number of iterations",
                    type=int, default=1)
parser.add_argument("-r", "--regions", help="Number of regions",
                    type=int, default=1)
parser.add_argument("-x", "--nx", help="Total number of x points (global)",
                    type=int, default=100)
parser.add_argument("-t", "--nt", help="Total number of t points (global)",
                    type=int, default=100)
parser.add_argument("-o", "--overlap", help="overlap",
                    type=int, default=2)
args = parser.parse_args()


comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

n_itr = args.steps
n_reg  = args.regions

reg = rank%n_reg
itr = int(rank/n_reg)

overlap = args.overlap
offset  = overlap*2
nt = args.nt
nx = args.nx

dt = 1./(nt-1)
dx = 2.*np.pi/(nx-1)

x0_full = np.sin(np.linspace(0, 2*np.pi, nx))
x0      = region_views(x0_full, n_reg, overlap)[reg]

A = one_d_heat_btcs(len(x0), dx, dt)

solver = ImplicitSolver1D(A)
r      = Region(nt, len(x0))
solver.x[:]    = x0
r.slices[0][:] = x0

has_left  = reg>0
has_right = reg<n_reg-1
has_prev_itr = itr>0
has_next_itr = itr<n_itr-1
    
right = rank+1
left  = rank-1
prev_itr = rank-n_reg
next_itr = rank+n_reg
    
if has_right:
    r.add_col(0, -1-offset)
if has_left:
    r.add_col(0, offset)

for _ in range(1):
    solver.x[:] = x0
    i_start = 1 - itr 
    
    for i in range(i_start, nt):
        send_requests = []
        if i>0:
            solver.left  = r.cols[0][0][i]
            solver.right = r.cols[0][-1][i]
            solver.solve()
            r.update_cols(i, solver.x)
            
            if has_right:
                rr = comm.Isend(r.cols[0][-1-offset][i:i+1], dest=right)
                send_requests.append(rr)
            if has_left:
                rl = comm.Isend(r.cols[0][offset][i:i+1], dest=left)
                send_requests.append(rl)
                
            if has_right:
                comm.Recv(r.cols[0][-1][i:i+1], source=right)
            if has_left:
                comm.Recv(r.cols[0][0][i:i+1], source=left)
                
            if has_next_itr:
                #print rank, i, "next"
                rn1 = comm.Isend(r.cols[0][0][i:i+1],  dest=next_itr, tag=1)
                rn2 = comm.Isend(r.cols[0][-1][i:i+1], dest=next_itr, tag=2)
                send_requests.append(rn1)
                send_requests.append(rn2)
            
        if i>=0:
            if has_prev_itr and i+1<nt:
                comm.Recv(r.cols[0][0][i+1:i+2],  source=prev_itr, tag=1)
                comm.Recv(r.cols[0][-1][i+1:i+2], source=prev_itr, tag=2)
                    
        MPI.Request.Waitall(send_requests)    
