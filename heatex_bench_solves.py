
import time
import argparse
import numpy as np
from mpi4py import MPI
from region import *
from recursive import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--steps", help="Number of measurements",
                    type=int, default=1)
parser.add_argument("-x", "--nx", help="Total number of x points (global)",
                    type=int, default=100)
parser.add_argument("-y", "--ny", help="Total number of y points (global)",
                    type=int, default=100)
parser.add_argument("-t", "--nt", help="Total number of time steps",
                    type=int, default=100)
parser.add_argument("--reg-x", help="Number of regions in x",
                    type=int, default=-1)
parser.add_argument("--reg-y", help="Number of regions in y",
                    type=int, default=-1)
parser.add_argument("--time", help="Print the average of elapsed run times",
                    action="store_true")
args = parser.parse_args()

#######################################
# Set up MPI
comm  = MPI.COMM_WORLD
rank  = comm.rank
size  = comm.size

n_reg_x = args.reg_x
n_reg_y = args.reg_y

cart   = comm.Create_cart((n_reg_y, n_reg_x))
# The location of this nodes region
ry, rx = cart.Get_coords(rank) 

has_left  = rx>0
has_right = rx<n_reg_x-1
has_north = ry<n_reg_y-1
has_south = ry>0

#####################################    
# Set up init vals and solver
nt = args.nt
nx = args.nx
ny = args.ny

x_max = 3.*np.pi
y_max = 3.*np.pi
dt = 1./(nt-1)
dx = x_max/(nx-1)
dy = y_max/(ny-1)

x0_full, y0_full = np.meshgrid(np.linspace(0, x_max, nx), np.linspace(0, y_max, ny))
f0_full  = np.sin(x0_full)*np.sin(y0_full)

f0 = region_views_2d(f0_full, n_reg_y, n_reg_x)[ry][rx]
x0 = region_views_2d(x0_full, n_reg_y, n_reg_x)[ry][rx]
y0 = region_views_2d(y0_full, n_reg_y, n_reg_x)[ry][rx]

# Build solver
solver = ImplicitSolver2DRec(dt, f0.shape[0], dy, 
                                 f0.shape[1], dx, 
                                 has_left, has_right,
                                 has_north, has_south)

solver.x[:] = f0

times = []
for _ in range(args.steps):
    start = time.clock()
    solver.solve()
    end = time.clock()
    times.append(end-start)

    
all_times = comm.gather(times, root=0)

if rank==0:
    for t in all_times:
        print np.mean(t)

    
