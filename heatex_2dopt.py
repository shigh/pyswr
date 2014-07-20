
import time
import argparse
import numpy as np
from mpi4py import MPI
from pyswr.region import *
from pyswr.recursive import *
from pyswr.utils import *
from pyswr.swr import *

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

cart   = comm.Create_cart((n_reg_y, n_reg_x))
# The location of this nodes region
ry, rx = cart.Get_coords(rank) 

# Determine which adjacent nodes exist
has_left  = rx>0
has_right = rx<n_reg_x-1
has_north = ry<n_reg_y-1
has_south = ry>0


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

x_range = partition_domain(nx, n_reg_x, 0)[rx]
y_range = partition_domain(ny, n_reg_y, 0)[ry]
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
region = RecBoundarySet(nt, solver.x.shape)
region.slices[0][:] = f0

#############################
# Schwartz waveform Iteration

start = time.clock()

swr_opt_heat(MPI, comm, (n_reg_y, n_reg_x), region, solver, args.steps)
comm.Barrier()

end = time.clock()
elapsed_time = end - start

if args.time:

    all_times = comm.gather(elapsed_time, root=0)

    if rank==0:
        print "Max Runtime: %f" % (np.max(all_times),)

if args.error:    

    all_last_vals = comm.gather(((ry, rx), region.slices[nt-1]), root=0)

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
        errors = [np.max(np.abs(exact_views[z[0]][z[1]]-v0))
                  for (z, v0) in all_last_vals]

        print np.max(errors)
                            
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

