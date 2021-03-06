import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from collections import Iterable
import argparse

def one_d_poisson(n, h, include_boundary=True):
    """
    Returns an nxn matrix that solves the difference 
    equations with fixed boundaries
    """
    a = np.zeros((n,n))
    if include_boundary:
        np.fill_diagonal(a[1:-1,1:-1], 2.)
        np.fill_diagonal(a[1:-1,:-1], -1.)
        np.fill_diagonal(a[1:-1,2:], -1.)
        a = a/h**2
        a[0,0]=a[-1,-1]=1.
        return  a
    else:
        np.fill_diagonal(a, 2.)
        np.fill_diagonal(a[:-1,1:], -1.)
        np.fill_diagonal(a[1:,:-1], -1.)
        a = a/h**2
        return  a

def one_d_heat_btcs(n, dx, dt):
    """Au^{n+1} = u^n for heatequation

    Backward time, centered difference space
    """
    a = np.zeros((n,n), dtype=np.double)
    beta = dt/(dx**2)
    np.fill_diagonal(a[1:-1,1:-1], 1+2.*beta)
    np.fill_diagonal(a[1:-1,:-1], -beta)
    np.fill_diagonal(a[1:-1,2:],  -beta)
    a[0,0]=a[-1,-1]=1.
    return  a


def two_d_heat_btcs(dt, ny, dy, nx, dx, as_csr=True):
    """Au^{n+1} = u^n for heatequation

    Backward time, centered difference space
    """
    kx = dt/(dx**2)
    ky = dt/(dy**2)

    N = nx*ny
    A = sparse.lil_matrix((N, N), dtype=np.double)
    
    for i in range(ny):
        for j in range(nx):

            # row/col for point
            m = j+i*nx 
            
            # Either the boundary or the I part of the equation
            A[m, m] = 1
            
            # Differentials
            if not(i==0 or j==0 or i==ny-1 or j==nx-1):
                # x-direction
                A[m, m]   -= -2*kx
                A[m, m+1] -= 1*kx
                A[m, m-1] -= 1*kx
                
                # y-direction
                A[m, m]    -= -2*ky
                A[m, m+nx] -= 1*ky
                A[m, m-nx] -= 1*ky

    if as_csr:
        return A.tocsr()
    else:
        return A


def three_d_heat_btcs(dt, nz, dz, ny, dy, nx, dx):
    """Au^{n+1} = u^n for heatequation

    Backward time, centered difference space
    """
    kx = dt/(dx**2)
    ky = dt/(dy**2)
    kz = dt/(dz**2)

    N = nx*ny*nz
    A = sparse.lil_matrix((N, N), dtype=np.double)
    
    for k in range(nz):
        for i in range(ny):
            for j in range(nx):
                
                # row/col for point
                m = j+i*nx+k*nx*ny 
                
                # Either the boundary or the I part of the equation
                A[m, m] = 1
                
                # Differentials
                if not(k==0    or i==0    or j==0 or
                       k==nz-1 or i==ny-1 or j==nx-1):
                    # x-direction
                    A[m, m]   -= -2*kx
                    A[m, m+1] -= 1*kx
                    A[m, m-1] -= 1*kx
                    
                    # y-direction
                    A[m, m]    -= -2*ky
                    A[m, m+nx] -= 1*ky
                    A[m, m-nx] -= 1*ky
                    
                    # z-direction
                    A[m, m]       -= -2*kz
                    A[m, m+nx*ny] -= 1*kz
                    A[m, m-nx*ny] -= 1*kz
    
    return A.tocsr()
    

def partition_domain(N, n, k):
    domain_size = int(round((float(N+(n-1)*(1+k)))/(float(n))))
    domains = []
    last_end = k
    for i in range(n):
        domains.append((last_end-k,
                        last_end-k+domain_size))
        last_end = last_end-k+domain_size-1

    domains[-1] = (domains[-1][0], N)    
    return domains


def region_views(array, n, k):
    N = array.shape[-1]
    domains = partition_domain(N, n, k)
    views = []
    for i in range(n):
        a, b = domains[i]
        views.append(array[a:b])
    
    return views


def build_domains(init_vals, Nt, n, k):
    
    N = len(init_vals)
    
    domains = partition_domain(N, n, k)
    Exa = [np.zeros((Nt, b-a))   for a,b in domains]
    Hya = [np.zeros((Nt, b-a-1)) for a,b in domains]
    
    for i in range(n):
        a, b = domains[i]
        Exa[i][0,:] = init_vals[a:b]

    return (Exa, Hya)
    

def region_views_2d(arr, n_regions_y, n_regions_x, overlap_y=0, overlap_x=0):
    """Views of all regions in an array.
    """    
    ny, nx = arr.shape
    rsi_y = partition_domain(ny, n_regions_y, overlap_y)
    rsi_x = partition_domain(nx, n_regions_x, overlap_x)
    
    views = [[arr[rsi_y[j][0]:rsi_y[j][1], rsi_x[i][0]:rsi_x[i][1]]
              for i in range(n_regions_x)]
              for j in range(n_regions_y)]

    return views
    

def array_avg_denom(n_steps, n_arrays, overlap):
    """An array of the number of overlap counts.
    """
    base = np.zeros(n_steps, dtype=np.float)
    for rv in region_views(base, n_arrays, overlap):
        rv += 1
    return base


def combine_arrays(arrays, n_steps, overlap=0):
    """Average together arrays in domain decomp

    Expects them to be in the same order as
    region views returns
    """

    n_arrays = len(arrays)
    denom = array_avg_denom(n_steps, n_arrays, overlap)

    avg_array = np.zeros(n_steps, dtype=arrays[0].dtype)
    avg_views = region_views(avg_array, n_arrays, overlap)

    for a, b in zip(avg_views, arrays):
        a[:] = a + b

    return avg_array/denom
    

def as_tuple(a):        
    """
    Accepts iterable or non-iterable argument
    """

    if isinstance(a, Iterable):
        return tuple(a)
    else:
        return (a,)
    

# Setup argument parser        
parser1d = argparse.ArgumentParser()
parser1d.add_argument("-s", "--steps", help="Number of iterations",
                      type=int, default=1)
parser1d.add_argument("-x", "--nx", help="Total number of x points (global)",
                      type=int, default=100)
parser1d.add_argument("-t", "--nt", help="Total number of t points (global)",
                      type=int, default=100)
parser1d.add_argument("-o", "--overlap", help="overlap",
                      type=int, default=2)
parser1d.add_argument("-r", "--regions", help="Number of regions",
                      type=int, default=1)
parser1d.add_argument("--plot", help="Plot Results",
                      action="store_true")
parser1d.add_argument("--error", help="Print the error",
                      action="store_true")
parser1d.add_argument("--time", help="Print the average of elapsed run times",
                      action="store_true")        


parser2d = argparse.ArgumentParser()
parser2d.add_argument("-s", "--steps", help="Number of iterations",
                      type=int, default=1)
parser2d.add_argument("-x", "--nx", help="Total number of x points (global)",
                      type=int, default=100)
parser2d.add_argument("-y", "--ny", help="Total number of y points (global)",
                      type=int, default=100)
parser2d.add_argument("-t", "--nt", help="Total number of t points (global)",
                      type=int, default=100)
parser2d.add_argument("--reg-x", help="Number of regions in x",
                      type=int, default=-1)
parser2d.add_argument("--reg-y", help="Number of regions in y",
                      type=int, default=-1)
parser2d.add_argument("--plot", help="Plot Results",
                      action="store_true")
parser2d.add_argument("--error", help="Print the error",
                      action="store_true")
parser2d.add_argument("--time", help="Print the average of elapsed run times",
                      action="store_true")

