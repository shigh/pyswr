
import time
import argparse
import numpy as np
from mpi4py import MPI
from pyswr.region import *
from pyswr.recursive import *
from pyswr.utils import *
from pyswr.itertable import *

args = parser2d.parse_args()

#######################################
# Set up MPI
comm  = MPI.COMM_WORLD
rank  = comm.rank
size  = comm.size

########################################
# Set up MPI cart and location variables

n_reg_x = args.reg_x
n_reg_y = args.reg_y
n_itr = int(size/(n_reg_x*n_reg_y))

cart   = comm.Create_cart((n_itr, n_reg_y, n_reg_x))
# The location of this nodes region
itr, ry, rx = cart.Get_coords(rank)
it_table = IterTable(cart, args.nt, args.steps)

# Determine which adjacent nodes exist
# TODO Abstract all of this nasty location code into a
#      dedicated object that can handle ndims
has_left  = rx>0
has_right = rx<n_reg_x-1
has_north = ry<n_reg_y-1
has_south = ry>0
has_prev_itr = itr>0       or itr==0
has_next_itr = itr<n_itr-1 or itr==n_itr-1

left = right = north = south    = None
next_itr_left  = next_itr_right = None
next_itr_north = next_itr_south = None
prev_itr_left  = prev_itr_right = None
prev_itr_north = prev_itr_south = None

if itr!=0 and itr!=n_itr-1:
    if has_left:
        prev_itr_left  = cart.Get_cart_rank((itr-1, ry, rx-1))
        next_itr_left  = cart.Get_cart_rank((itr+1, ry, rx-1))
    if has_right:
        prev_itr_right = cart.Get_cart_rank((itr-1, ry, rx+1))
        next_itr_right = cart.Get_cart_rank((itr+1, ry, rx+1))
    if has_north:
        prev_itr_north = cart.Get_cart_rank((itr-1, ry+1, rx))
        next_itr_north = cart.Get_cart_rank((itr+1, ry+1, rx))
    if has_south:
        prev_itr_south = cart.Get_cart_rank((itr-1, ry-1, rx))
        next_itr_south = cart.Get_cart_rank((itr+1, ry-1, rx))
elif itr==0:
    if has_left:
        prev_itr_left  = cart.Get_cart_rank((n_itr-1, ry, rx-1))
        next_itr_left  = cart.Get_cart_rank((itr+1, ry, rx-1))
    if has_right:
        prev_itr_right = cart.Get_cart_rank((n_itr-1, ry, rx+1))
        next_itr_right = cart.Get_cart_rank((itr+1, ry, rx+1))
    if has_north:
        prev_itr_north = cart.Get_cart_rank((n_itr-1, ry+1, rx))
        next_itr_north = cart.Get_cart_rank((itr+1, ry+1, rx))
    if has_south:
        prev_itr_south = cart.Get_cart_rank((n_itr-1, ry-1, rx))
        next_itr_south = cart.Get_cart_rank((itr+1, ry-1, rx))                
elif itr==n_itr-1:
    if has_left:
        next_itr_left  = cart.Get_cart_rank((0, ry, rx-1))
        prev_itr_left  = cart.Get_cart_rank((itr-1, ry, rx-1))
    if has_right:
        next_itr_right = cart.Get_cart_rank((0, ry, rx+1))
        prev_itr_right = cart.Get_cart_rank((itr-1, ry, rx+1))
    if has_north:
        next_itr_north = cart.Get_cart_rank((0, ry+1, rx))
        prev_itr_north = cart.Get_cart_rank((itr-1, ry+1, rx))
    if has_south:
        next_itr_south = cart.Get_cart_rank((0, ry-1, rx))
        prev_itr_south = cart.Get_cart_rank((itr-1, ry-1, rx))


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
# One complete iteration per node for now
# TODO Arbitrary number of iterations per node
start = time.clock()

# Reset solver for next iteration
solver.x[:] = region.slices[0]

# Keep track of final error for reporting
#(it_table.last_itr, region.slices[nt-1].copy())]
last_slice_vals = []
def update_last_slice_vals():
    if args.error and it_table.t==(nt-1):
        last_slice_vals.append((it_table.location[1:], it_table.itr, region.slices[nt-1].copy()))

# Apply solver over each time step
while not it_table.has_finished:

    t = it_table.t

    # Reset for next iteration
    if it_table.reset_solver:
        solver.x[:] = region.slices[0]

    send_requests = []        

    if t>0:
        solver.g = [[region.g[0][0][t], region.g[0][-1][t]],
                    [region.g[1][0][t], region.g[1][-1][t]]]
        solver.solve()
        region.update_cols(t, solver.x)
        update_last_slice_vals()

    # Communicate with adjacent regions
    if it_table.next_active and t>0:
        if has_right:
            rnr = comm.Isend(region.send_g(1, -1)[t:t+1], dest=next_itr_right)
            send_requests.append(rnr)
        if has_left:
            rnl = comm.Isend(region.send_g(1, 0)[t:t+1], dest=next_itr_left)
            send_requests.append(rnl)
        if has_north:
            rnn = comm.Isend(region.send_g(0, -1)[t:t+1], dest=next_itr_north)
            send_requests.append(rnn)
        if has_south:
            rns = comm.Isend(region.send_g(0, 0)[t:t+1], dest=next_itr_south)
            send_requests.append(rns)

    if it_table.prev_active:
        tp = it_table.t_prev
        if has_right:
            comm.Recv(region.g[1][-1][tp:tp+1], source=prev_itr_right)
        if has_left:
            comm.Recv(region.g[1][0][tp:tp+1], source=prev_itr_left)
        if has_north:
            comm.Recv(region.g[0][-1][tp:tp+1], source=prev_itr_north)
        if has_south:
            comm.Recv(region.g[0][0][tp:tp+1], source=prev_itr_south)

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

    all_last_vals = comm.gather(last_slice_vals, root=0)

    if rank==0:

        # Build init vals
        x0_full, y0_full = np.meshgrid(np.linspace(0, x_max, nx), np.linspace(0, y_max, ny))
        f0_full  = np.sin(x0_full)*np.sin(y0_full)

        # Flatten all_last_vals
        vals = []
        for r in all_last_vals:
            for c in r:
                vals.append(c)

        # Reshape by location
        locations = []
        for v in vals:
            if not v[0] in locations:
                locations.append(v[0])

        vals = [sorted([v for v in vals if v[0]==c],
                       key=lambda x:x[1])
                for c in locations]

        s = ImplicitSolver2DRec(dt, ny, dy, nx, dx, False, False, False, False)
        s.x[:] = f0_full
        for _ in range(1, nt):
            s.solve()

        exact_views = region_views_2d(s.x, n_reg_y, n_reg_x)
        errors = [[np.max(np.abs(exact_views[z[0]][z[1]]-v0))
                   for (z,_,v0) in v]
                   for v in vals]

        errors = np.array(errors)

        for i in range(len(errors[0])):
            print "Itr", i+1, ":", np.max(errors[:, i])

if args.plot:

    all_x_vals = comm.gather(x0, root=0)
    all_y_vals = comm.gather(y0, root=0)
    all_f_vals = comm.gather((it_table.last_itr, region.slices[nt-1]), root=0)

    if rank==0:

        all_f_vals = [f for (i, f) in all_f_vals
                      if i==(args.steps-1)]
        
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d

        fig = plt.figure()
        fig.set_size_inches(14, 10)
        ax = fig.add_subplot(111, projection='3d')
        cs, rs = 1, 1

        for i in range(n_reg_y*n_reg_x):
            ax.plot_wireframe(all_x_vals[i], all_y_vals[i], all_f_vals[i], 
                              cstride=cs, rstride=rs, color='r')

        if args.error:
            ax.plot_wireframe(x0_full, y0_full, s.x,
                              cstride=cs, rstride=rs)
            
        #ax.plot_wireframe(x0_full, y0_full, expected, color='r', cstride=cs, rstride=rs)
        #ax.plot_wireframe(x0_full, y0_full, x, color='r', cstride=cs, rstride=rs)
        plt.show()

