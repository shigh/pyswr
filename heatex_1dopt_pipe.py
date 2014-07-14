
import time
import argparse
import numpy as np
from mpi4py import MPI
from pyswr.region import *
from pyswr.recursive import *
from pyswr.utils import *
from pyswr.itertable import *
from pyswr.swr import *

args = parser1d.parse_args()

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
has_left     = reg>0
has_right    = reg<n_reg-1

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

it_table = pswr_opt_heat(MPI, comm, (n_itr, n_reg), region, solver, args.steps)        

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
        itr_vals = np.unique([e[0] for e in all_itr_error
                              if e[0]>=0])
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
