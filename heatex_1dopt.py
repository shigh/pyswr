
import time
import argparse
import numpy as np
from mpi4py import MPI
from region import *
from recursive import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--steps", help="Number of iterations",
                    type=int, default=1)
parser.add_argument("-x", "--nx", help="Total number of x points (global)",
                    type=int, default=100)
parser.add_argument("-t", "--nt", help="Total number of t points (global)",
                    type=int, default=100)
parser.add_argument("--plot", help="Plot Results",
                    action="store_true")
parser.add_argument("--error", help="Print the error",
                    action="store_true")
parser.add_argument("--time", help="Print the average of elapsed run times",
                    action="store_true")
args = parser.parse_args()

comm  = MPI.COMM_WORLD
rank  = comm.rank
size  = comm.size
n_reg = size

nt = args.nt
nx = args.nx

x_max = 3.*np.pi
dt = 1./(nt-1)
dx = x_max/(nx-1)

x0_full  = np.linspace(0, x_max, nx)
f0_full  = np.sin(x0_full)

f0 = region_views(f0_full, n_reg, 0)[rank]
x0 = region_views(x0_full, n_reg, 0)[rank]

has_left  = rank>0
has_right = rank<n_reg-1

right = rank+1
left  = rank-1

# Build solver and region
solver = ImplicitSolver1DRec(dt, len(f0), dx, has_left, has_right)
region = RecRegion(nt, len(solver.x))
region.slices[0][:] = f0

last_slice_vals = []
def update_last_slice_vals(t, step):
    if args.error and t==(nt-1):
        last_slice_vals.append((rank, step, region.slices[nt-1].copy()))

start = time.clock()
for step in range(args.steps):

    # Reset solver for next iteration
    solver.x[:] = region.slices[0]
    
    # Apply solver over each time step
    for t in range(1, nt):
        solver.g = [region.g[0][0][t], region.g[0][-1][t]]
        solver.solve()
        region.update_cols(t, solver.x)
        update_last_slice_vals(t, step)

    send_requests = []
    if has_right:
        rr = comm.Isend(region.send_g(0, -1), dest=right)
        send_requests.append(rr)
    if has_left:
        rl = comm.Isend(region.send_g(0, 0), dest=left)
        send_requests.append(rl)
        
    if has_right:
        comm.Recv(region.g[0][-1], source=right)
    if has_left:
        comm.Recv(region.g[0][0], source=left)
        
    MPI.Request.Waitall(send_requests)

comm.Barrier()    
end = time.clock()
elapsed_time = end - start

if args.time:

    all_times = comm.gather(elapsed_time, root=0)

    if rank==0:
        print "Max Runtime: %f" % (np.max(all_times),)
    
if args.error:    

    all_last_vals = comm.gather(last_slice_vals, root=0)

    if rank==0:

        s = ImplicitSolver1DRec(dt, len(f0_full), dx, False, False)
        s.x[:] = f0_full
        for _ in range(1, nt):
            s.solve()

        exact_views = region_views(s.x, n_reg, 0)
        errors = [[np.max(np.abs(exact_views[z]-v0))
                   for (z, _, v0) in v]
                   for v in all_last_vals]

        errors = np.array(errors)

        for i in range(len(errors[0])):
            print "Itr", i+1, ":", np.max(errors[:, i])
        
if args.plot:

    all_last_vals = comm.gather(last_slice_vals, root=0)
    all_x_vals = comm.gather(x0, root=0)
    all_f_vals = comm.gather(region.slices[nt-1], root=0)

    if rank==0:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        for i, j in zip(all_x_vals, all_f_vals):
            ax.plot(i, j)

        plt.show()
