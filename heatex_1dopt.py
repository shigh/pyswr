
import time
import argparse
import numpy as np
from mpi4py import MPI
from pyswr.region import *
from pyswr.recursive import *
from pyswr.utils import *
from pyswr.swr import *

args = parser1d.parse_args()

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

# Build solver and region
solver = ImplicitSolver1DRec(dt, len(f0), dx, has_left, has_right)
region = RecBoundarySet(nt, len(solver.x))
region.slices[0][:] = f0

# last_slice_vals = []
# def update_last_slice_vals(t, step):
#     if args.error and t==(nt-1):
#         last_slice_vals.append((rank, step, region.slices[nt-1].copy()))

start = time.clock()

swr_opt_heat(MPI, comm, n_reg, region, solver, args.steps)
comm.Barrier()

end = time.clock()
elapsed_time = end - start

if args.time:

    all_times = comm.gather(elapsed_time, root=0)

    if rank==0:
        print "Max Runtime: %f" % (np.max(all_times),)
    
if args.error:    

    # all_last_vals = comm.gather(last_slice_vals, root=0)
    all_last_vals = comm.gather((rank, region.slices[nt-1]), root=0)

    if rank==0:

        s = ImplicitSolver1DRec(dt, len(f0_full), dx, False, False)
        s.x[:] = f0_full
        for _ in range(1, nt):
            s.solve()

        exact_views = region_views(s.x, n_reg, 0)
        errors = [np.max(np.abs(exact_views[z]-v0))
                  for (z, v0) in all_last_vals]

        errors = np.array(errors)

        print np.max(errors)
        
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
