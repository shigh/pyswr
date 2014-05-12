
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
parser.add_argument("-y", "--ny", help="Total number of y points (global)",
                    type=int, default=100)
parser.add_argument("-t", "--nt", help="Total number of t points (global)",
                    type=int, default=100)
parser.add_argument("--reg-x", help="Number of regions in x",
                    type=int, default=-1)
parser.add_argument("--reg-y", help="Number of regions in y",
                    type=int, default=-1)
parser.add_argument("--plot", help="Plot Results",
                    action="store_true")
parser.add_argument("--error", help="Print the error",
                    action="store_true")
parser.add_argument("--time", help="Print the average of elapsed run times",
                    action="store_true")
args = parser.parse_args()

#######################################
# Set up MPI
comm  = MPI.COMM_WORLD
rank  = comm.rank
size  = comm.size

########################################
# Set up MPI cart and location variables

n_reg_x = args.reg_x
n_reg_y = args.reg_y

cart   = comm.Create_cart((n_reg_y, n_reg_x))
# The location of this nodes region
ry, rx = cart.Get_coords(rank) 

# Determine which adjacent nodes exist
has_left  = rx>0
has_right = rx<n_reg_x-1
has_north = ry<n_reg_y-1
has_south = ry>0
left = right = north = south = None

# Get ranks of adjacent nodes
if has_left:
    left = cart.Get_cart_rank((ry, rx-1))
if has_right:
    right = cart.Get_cart_rank((ry, rx+1))
if has_north:
    north = cart.Get_cart_rank((ry+1, rx))
if has_south:
    south = cart.Get_cart_rank((ry-1, rx))


#####################################    
# Set up init vals, region and solver
nt = args.nt
nx = args.nx
ny = args.ny

x_max = 3.*np.pi
y_max = 3.*np.pi
dt = 1./(nt-1)
dx = x_max/(nx-1)
dy = y_max/(ny-1)

x_range = region_slice_index(nx, n_reg_x, 0)[rx]
y_range = region_slice_index(ny, n_reg_y, 0)[ry]
x_start, x_end = x_range[0]*dx, (x_range[1]-1)*dx
y_start, y_end = y_range[0]*dy, (y_range[1]-1)*dy
x_points = x_range[1]-x_range[0]
y_points = y_range[1]-y_range[0]

X, Y = np.meshgrid(np.linspace(x_start, x_end, x_points), np.linspace(y_start, y_end, y_points))
f0 = np.sin(X)*np.sin(Y)

# Build solver
solver = ImplicitSolver2DRec(dt, f0.shape[0], dy, 
                                 f0.shape[1], dx, 
                                 has_left, has_right,
                                 has_north, has_south)
                                 
# Build region
region = RecRegion(nt, solver.x.shape)
region.slices[0][:] = f0

#############################
# Schwartz waveform Iteration
last_slice_vals = []
def update_last_slice_vals(t, step):
    if args.error and t==(nt-1):
        last_slice_vals.append(((ry, rx), step, region.slices[nt-1].copy()))

start = time.clock()
for step in range(args.steps):

    # Reset solver for next iteration
    solver.x[:] = region.slices[0]

    # Apply solver over each time step
    for t in range(1, nt):
        # TODO Convert this to a nested list comprehension
        solver.g = [[region.g[0][0][t], region.g[0][-1][t]],
                    [region.g[1][0][t], region.g[1][-1][t]]]
        solver.solve()
        region.update_cols(t, solver.x)
        update_last_slice_vals(t, step)


    # Communicate with adjacent regions
    send_requests = []
    if has_right:
        rr = comm.Isend(region.send_g(1, -1), dest=right)
        send_requests.append(rr)
    if has_left:
        rl = comm.Isend(region.send_g(1, 0), dest=left)
        send_requests.append(rl)
    if has_north:
        rn = comm.Isend(region.send_g(0, -1), dest=north)
        send_requests.append(rn)
    if has_south:
        rs = comm.Isend(region.send_g(0, 0), dest=south)
        send_requests.append(rs)
        
    if has_right:
        comm.Recv(region.g[1][-1], source=right)
    if has_left:
        comm.Recv(region.g[1][0], source=left)
    if has_north:
        comm.Recv(region.g[0][-1], source=north)
    if has_south:
        comm.Recv(region.g[0][0], source=south)
        
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

        x0_full, y0_full = np.meshgrid(np.linspace(0, x_max, nx), np.linspace(0, y_max, ny))
        f0_full  = np.sin(x0_full)*np.sin(y0_full)
        
        s = ImplicitSolver2DRec(dt, f0_full.shape[0], dy,
                                    f0_full.shape[1], dx,
                                False, False, False, False)
        s.x[:] = f0_full
        for _ in range(1, nt):
            s.solve()

        exact_views = region_views_2d(s.x, n_reg_y, n_reg_x)
        errors = [[np.max(np.abs(exact_views[z[0]][z[1]]-v0))
                   for (z, _, v0) in v]
                   for v in all_last_vals]

        errors = np.array(errors)

        for i in range(len(errors[0])):
            print "Itr", i+1, ":", np.max(errors[:, i])
                            
if args.plot:

    all_x_vals = comm.gather(x0, root=0)
    all_y_vals = comm.gather(y0, root=0)
    all_f_vals = comm.gather(region.slices[nt-1], root=0)

    if rank==0:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d

        fig = plt.figure()
        fig.set_size_inches(14, 10)
        ax = fig.add_subplot(111, projection='3d')
        cs, rs = 1, 1

        for i in range(n_reg_y*n_reg_x):
            ax.plot_wireframe(all_x_vals[i], all_y_vals[i], all_f_vals[i], 
                              cstride=cs, rstride=rs,
                              color='r')

        if args.error:
            ax.plot_wireframe(x0_full, y0_full, s.x,
                              cstride=cs, rstride=rs,
                              color='b')
        
        plt.show()

