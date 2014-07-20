
import argparse
import numpy as np
from mpi4py import MPI
from pyswr.region import *
from pyswr.utils import *
from pyswr.swr import *

args = parser1d.parse_args()

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

overlap = args.overlap
nt = args.nt
nx = args.nx

dt = 1./(nt-1)
dx = 2.*np.pi/(nx-1)

x_vals_full = np.linspace(0, 2*np.pi, nx)
x_vals      = region_views(x_vals_full, size, overlap)[rank]
f0          = np.sin(x_vals)

A = one_d_heat_btcs(len(f0), dx, dt)
solver = ImplicitSolver1D(A)
region = BoundarySet(nt, len(f0))
solver.x[:] = f0
region.slices[0][:] = f0


swr_1d_heat(MPI, comm, size, region, solver, f0, args.steps, overlap)


if args.error:    
    expected = np.exp(-1.)*f0    
    error = np.max(np.abs(region.slices[nt-1] - expected))
    all_error = comm.gather(error, root=0)

    if rank==0:
        print "Avg: %f Min: %f Max: %f" % (np.mean(all_error),
                                           np.min(all_error),
                                           np.max(all_error))

if args.plot:

    all_x_vals = comm.gather(x_vals, root=0)
    all_f_vals = comm.gather(region.slices[nt-1], root=0)

    if rank==0:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        for i, j in zip(all_x_vals, all_f_vals):
            ax.plot(i, j)

        plt.show()
