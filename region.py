import numpy as np
from utils import *

class ImplicitSolver1D(object):
    
    def solver(self, A, tmp):
        return np.linalg.solve(A, tmp)
    
    def __init__(self, A):
        self.A = A
        self.x = np.zeros(len(self.A), dtype=np.double)
        self.left  = 0.
        self.right = 0.
        
    def solve(self):
        tmp       = self.x.copy()
        tmp[0]    = self.left
        tmp[-1]   = self.right
        self.x[:] = self.solver(self.A, tmp)


class Region(object):
    """Data structure for time dependent pde's
    """
    
    def __init__(self, nt, dims):
        self.nt   = nt
        #self.dims = (nx,)
        self.dims = as_tuple(dims)
        self.n_dims = len(self.dims)
        # time slices
        self.slices = {}
        # init vals slice
        self.slices[0]    = self.__empty_slice
        # final vals slice
        self.slices[nt-1] = self.__empty_slice

        # Columns are stored in a list of dictionarys
        # Set the outer columns
        self.cols = []
        for i in range(self.n_dims):
            # left and right cols
            self.cols.append({})
            self.add_col(i, 0)
            self.add_col(i, -1)

        # Used in get_col_slice to simplify the code
        self.__full_slices = [slice(d) for d in self.dims]

    def add_col(self, dim, col):
        self.cols[dim][col] = self.empty_col(dim)
        
    @property
    def __empty_slice(self):
        return np.zeros(self.dims, dtype=np.double)

    def empty_col(self, dim):
        col_dims = [self.nt]
        for i in range(self.n_dims):
            if i!=dim:
                col_dims.append(self.dims[i])

        return np.zeros(col_dims, dtype=np.double)
        
    def add_slice(self, time_step):
        self.slices[time_step] = self.__empty_slice

    def __get_col_slice(self, array, dim, col):
        """Extract the boundary for col in dimension dim
        """
        col_slice_dims = []
        for i in range(self.n_dims):
            if i==dim:
                col_slice_dims.append(col)
            else:
                col_slice_dims.append(self.__full_slices[i])

        return array[col_slice_dims]

    def update_cols(self, time_step, array):
        
        # set col values
        #for d in range(len(self.cols)):
        for dim in range(self.n_dims):
            col = self.cols[dim]
            for k in col.keys():
                col[k][time_step] = self.__get_col_slice(array, dim, k)
            
        # set slice
        if time_step in self.slices:
            self.slices[time_step][:] = array

