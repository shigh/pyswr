import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from collections import Iterable

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
    
def boundary_points(n_steps, n_regions):
    """Location of boundary interfaces if overlap=0
    """
    k = int(n_steps/n_regions)
    points = [k*i for i in range(1, n_regions)]
    points = [1] + points + [n_steps]
    return [p-1 for p in points]
    #return np.array(points, np.int) - 1

def region_slice_index(n_steps, n_regions, overlap):
    """Slice indicies of regions in an array.
    """    
    min_point = 0
    max_point = n_steps-1
    bp = boundary_points(n_steps, n_regions)
            
    rsi = []
    for i in range(n_regions):
        start = bp[i]
        stop  = bp[i+1]
        if start > min_point:
            start -= overlap
        if stop  < max_point:
            stop  += overlap
        rsi.append((start, stop+1))
    return rsi

def region_views(arr, n_regions, overlap):
    """Views of all regions in an array.
    """    
    n_steps = len(arr)  
    rsi = region_slice_index(n_steps, n_regions, overlap) 
    
    views = []
    for idx in rsi:
        views.append(arr[idx[0]:idx[1]])
    return views

def region_views_2d(arr, n_regions_y, n_regions_x, overlap_y=0, overlap_x=0):
    """Views of all regions in an array.
    """    
    ny, nx = arr.shape
    rsi_y = region_slice_index(ny, n_regions_y, overlap_y)
    rsi_x = region_slice_index(nx, n_regions_x, overlap_x)
    
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
    
