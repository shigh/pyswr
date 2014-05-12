
import time
import argparse
import numpy as np
from mpi4py import MPI
from region import *
from recursive import *
from utils import *
from itertable import *

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--steps", help="Number of iterations",
                    type=int, default=1)
parser.add_argument("-r", "--regions", help="Number of regions",
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

# Set up MPI
comm  = MPI.COMM_WORLD
rank  = comm.rank
size  = comm.size

# Set up this nodes relative location
n_reg = args.regions
n_itr = int(size/n_reg)

cart   = comm.Create_cart((n_itr, n_reg))
# The location of this nodes region
itr, reg = cart.Get_coords(rank)
it_table = IterTable(cart, args.nt, args.steps)

# TODO Abstract all of this nasty location code into a
#      dedicated object that can handle ndims
has_left     = reg>0
has_right    = reg<n_reg-1
has_prev_itr = itr>0       or itr==0
has_next_itr = itr<n_itr-1 or itr==n_itr-1

right = left = next_itr = prev_itr = None
next_itr_left = next_itr_right     = None
prev_itr_left = prev_itr_right     = None


if itr!=0 and itr!=n_itr-1:
    if has_left:
        prev_itr_left  = cart.Get_cart_rank((itr-1, reg-1))
        next_itr_left  = cart.Get_cart_rank((itr+1, reg-1))
    if has_right:
        prev_itr_right = cart.Get_cart_rank((itr-1, reg+1))
        next_itr_right = cart.Get_cart_rank((itr+1, reg+1))            
elif itr==0:
    if has_left:
        prev_itr_left  = cart.Get_cart_rank((n_itr-1, reg-1))
        next_itr_left  = cart.Get_cart_rank((itr+1, reg-1))
    if has_right:
        prev_itr_right = cart.Get_cart_rank((n_itr-1, reg+1))
        next_itr_right = cart.Get_cart_rank((itr+1, reg+1))
elif itr==n_itr-1:
    if has_left:
        next_itr_left  = cart.Get_cart_rank((0, reg-1))
        prev_itr_left  = cart.Get_cart_rank((itr-1, reg-1))
    if has_right:
        next_itr_right = cart.Get_cart_rank((0, reg+1))
        prev_itr_right = cart.Get_cart_rank((itr-1, reg+1))


# Set up init vals    
nt = args.nt
nx = args.nx

x_max = 3.*np.pi
dt = 1./(nt-1)
dx = x_max/(nx-1)

x0_full  = np.linspace(0, x_max, nx)
f0_full  = np.sin(x0_full)

f0 = region_views(f0_full, n_reg, 0)[reg]
x0 = region_views(x0_full, n_reg, 0)[reg]

# Build solver and region
solver = ImplicitSolver1DRec(dt, len(f0), dx, has_left, has_right)
region = RecRegion(nt, len(solver.x))
region.slices[0][:] = f0

solver.x[:] = region.slices[0]

##############################################################################
# Waveform iteration #########################################################
##############################################################################

start = time.clock()

while not it_table.has_finished:

    t = it_table.t

    # Reset for next iteration
    if it_table.reset_solver:
        solver.x[:] = region.slices[0]

    send_requests = []

    if t>0:
        solver.g = [region.g[0][0][t], region.g[0][-1][t]]
        solver.solve()
        region.update_cols(t, solver.x)

    if it_table.next_active and t>0:
        # if it_table.location==[1,0]:
        #     print "Sending ", t
        if has_right:
            rnr = comm.Isend(region.send_g(0, -1)[t:t+1], dest=next_itr_right)
            send_requests.append(rnr)
        if has_left:
            rnl = comm.Isend(region.send_g(0, 0)[t:t+1], dest=next_itr_left)
            send_requests.append(rnl)

    if it_table.prev_active:
        tp = it_table.t_prev
        if it_table.location==[0,0]:
            print "Recv at ", t, " into ", tp
        if has_right:
            comm.Recv(region.g[0][-1][tp:tp+1], source=prev_itr_right)
        if has_left:
            comm.Recv(region.g[0][0][tp:tp+1], source=prev_itr_left)


    MPI.Request.Waitall(send_requests)

    it_table.advance()

# Wait for all processes to finish so the timing is meaningful        
comm.Barrier()        
end = time.clock()
elapsed_time = end - start

if args.time:

    all_times = comm.gather(elapsed_time, root=0)

    if rank==0:
        print "Max Runtime: %f" % (np.max(all_times),)

if args.error:    

    expected = np.exp(-1.)*f0    
    error = np.max(np.abs(region.slices[nt-1] - expected))
    all_itr_error = comm.gather((it_table.last_itr, error), root=0)

    if rank==0:
        itr_vals = np.unique([e[0] for e in all_itr_error])
        for i in itr_vals:
            avg = np.mean([e[1] for e in all_itr_error
                           if e[0]==i])
            print "itr: %i Avg: %f" % (i, avg)

if args.plot:

    all_x_vals = comm.gather(x0, root=0)
    all_f_vals = comm.gather(region.slices[nt-1], root=0)

    if rank==0:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        for i, j in zip(all_x_vals, all_f_vals):
            ax.plot(i, j)

        plt.show()
