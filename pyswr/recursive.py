"""
Implicit solvers for the heat equation with recursive bounary conditions
"""
import numpy as np
from pyswr.region import *
from pyswr.utils import *

class ImplicitSolver1DRec(object):
    """Recursive implicit boundary conditions for 1D heat equation
    """
    
    def __init__(self, dt, nx, dx, has_left, has_right, p=1):

        self.x  = np.zeros((nx,), dtype=np.double)
        self.nx = nx; self.dx = dx; self.dt = dt
        self.p  = p
        self.k  = dt/(dx**2)
        self.has_left  = has_left
        self.has_right = has_right
        self.g  = [0., 0.]

        self.build_A()
    
    def solver(self, A, tmp):
        """Solve Ax==tmp for x
        """
        return np.linalg.solve(A, tmp)
    
    def solve(self):
        """Solve using the current boundaries stored in g

        The solve happens inplace using x as the init vals and
        the g arrays as the recursive component of the boundary values. 
        """

        tmp = self.x.copy()
        
        if self.has_left:
            tmp[0] -= 2*self.k*self.dx*self.g[0]
        if self.has_right:
            tmp[-1] += 2*self.k*self.dx*self.g[1]
        
        self.x[:] = self.solver(self.A, tmp)

    def build_A(self):
        # A is dense for the 1D solver
        # TODO Make A sparse
        # TODO The order of arguments for the one d btcs matrix are not
        #      consistent with the rest of my code, update this later
        A  = one_d_heat_btcs(self.nx, self.dx, self.dt)
        if self.has_left:
            A[0,0] = 1+2*self.k+2*self.k*self.dx*self.p
            A[0,1] = -2*self.k
        if self.has_right:
            A[-1,-1] = 1+2*self.k+2*self.k*self.dx*self.p
            A[-1,-2] = -2*self.k

        self.A = A

    def send_g(self):
        """The g values to send left and right
        """
        # TODO Add dimension to sg to stay consistent with
        #      higher dimension solvers
        sg = [None, None]
        
        if self.has_left:
            sg[0] = self.g[0]+2*self.p*self.x[0]
        if self.has_right:
            sg[1] = self.g[1]-2*self.p*self.x[-1]
            
        return tuple(sg)

class ImplicitSolver2DRec(object):
    """Recursive implicit boundary conditions for 2D heat equation
    """
    
    # The order of the boundaries in init is not consistent with
    # the dimension ordering, but it is not worth changeing atm
    def __init__(self, dt, ny, dy, nx, dx, 
                 has_left, has_right,
                 has_north, has_south, p=1):
        
        # Store x as matrix, but make sure to flatten before solving
        self.x  = np.zeros((ny, nx), dtype=np.double)
        self.nx = nx; self.dx = dx; self.dt = dt
        self.ny = ny; self.dy = dy
        self.p  = p
        self.kx = dt/(dx**2)
        self.ky = dt/(dy**2)
        self.has_left = has_left; self.has_right = has_right
        self.has_north = has_north; self.has_south = has_south
        self.g = [[np.zeros(self.nx), np.zeros(self.nx)], 
                  [np.zeros(self.ny), np.zeros(self.ny)]]

        self.tmp = np.zeros((ny, nx), dtype=np.double)
        # Initialize solver matrix
        self.build_A()
    
    def solver(self, A, tmp):
        """Solve Ax==tmp for x
        """
        # A is constructed so that each row corresponds to one
        # element in x, so the first thing we need to do is flatten tmp
        # Assign to shape so that no copy is made
        # (see doc string for numpy.reshape)
        tmp.shape = self.ny*self.nx
        # Use the scipy sparse linear solver
        tmp[:] = splinalg.spsolve(self.A, tmp)
        tmp.shape = (self.ny, self.nx)
        return tmp
    
    def solve(self):
        """Solve using the current boundaries stored in g

        The solve happens inplace using x as the init vals and
        the g arrays as the recursive component of the boundary values. 
        """
        # s and tmp are dense, A is sparse
        # Use x as the init vals, but make a copy so we can change the
        # boundaries to use recursive conditions
        # TODO Code this without making a copy of x
        self.tmp[:] = self.x
        tmp = self.tmp

        # Useful temporaries
        kxdx2 = 2.*self.kx*self.dx
        kydy2 = 2.*self.ky*self.dy

        if self.has_left:
            tmp[1:-1,0]  = (self.x[1:-1,0]  - kxdx2*self.g[1][0][1:-1])
        if self.has_right:
            tmp[1:-1,-1] = (self.x[1:-1,-1] + kxdx2*self.g[1][-1][1:-1])
        if self.has_south:
            tmp[0,1:-1]  = (self.x[0,1:-1]  - kydy2*self.g[0][0][1:-1])
        if self.has_north:
            tmp[-1,1:-1] = (self.x[-1,1:-1] + kydy2*self.g[0][-1][1:-1])

        # Subtract out self.x on the boundaries because
        # of double counting
        if self.has_left and self.has_north:
            tmp[-1,0]  = self.x[-1,0]-kxdx2*self.g[1][0][-1]+kydy2*self.g[0][-1][0]
        if self.has_right and self.has_north:
            tmp[-1,-1] = self.x[-1,-1]+kxdx2*self.g[1][-1][-1]+kydy2*self.g[0][-1][-1]
        if self.has_left and self.has_south:
            tmp[0,0]   = self.x[0,0]-kxdx2*self.g[1][0][0]-kydy2*self.g[0][0][0]
        if self.has_right and self.has_south:
            tmp[0,-1]  = self.x[0, -1]+kxdx2*self.g[1][-1][0]-kydy2*self.g[0][0][-1]

        # Replace x with the updated values
        self.x[:] = self.solver(self.A, tmp)

    def build_A(self):

        # A is a scipy sparse lil matrix
        A = two_d_heat_btcs(self.dt, self.ny, self.dy, self.nx,
                            self.dx, as_csr=False)

        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        kx, ky = self.kx, self.ky
        p      = self.p
        # Useful temporaries
        alpha_x  = 1.+2.*kx+2.*kx*dx*p
        alpha_y  = 1.+2.*ky+2.*ky*dy*p
        alpha_xy = alpha_x + alpha_y - 1.

        # Set edges
        if self.has_left:
            for i in range(1, ny-1):
                m = i*nx
                A[m,m]    = alpha_x + 2.*ky
                A[m,m+1]  = -2.*kx
                A[m,m+nx] = -ky
                A[m,m-nx] = -ky
        if self.has_right:
            for i in range(1, ny-1):
                m = (i+1)*nx-1
                A[m,m]   = alpha_x + 2.*ky
                A[m,m-1] = -2.*kx
                A[m,m+nx] = -ky
                A[m,m-nx] = -ky
        if self.has_south:
            for i in range(1, nx-1):
                m = i
                A[m,m]    = alpha_y + 2.*kx
                A[m,m+nx] = -2.*ky
                A[m,m+1]  = -kx
                A[m,m-1]  = -kx
        if self.has_north:
            for i in range(1, nx-1):
                m = nx*(ny-1)+i
                A[m,m]    = alpha_y + 2.*kx
                A[m,m-nx] = -2.*ky
                A[m,m+1]  = -kx
                A[m,m-1]  = -kx

        # Set corners
        if self.has_left and self.has_north:
            m = nx*(ny-1)
            A[m,m] = alpha_xy
            A[m,m+1]  = -2.*kx
            A[m,m-nx] = -2.*ky
        if self.has_right and self.has_north:
            m = nx*ny-1
            A[m,m] = alpha_xy
            A[m,m-1]  = -2.*kx
            A[m,m-nx] = -2.*ky
        if self.has_left and self.has_south:
            m = 0
            A[m,m] = alpha_xy
            A[m,m+1]  = -2.*kx
            A[m,m+nx] = -2.*ky
        if self.has_right and self.has_south:
            m = nx-1
            A[m,m] = alpha_xy
            A[m,m-1]  = -2.*kx
            A[m,m+nx] = -2.*ky

        # Convert lil to csr
        self.A = A.tocsr()
                
        
    def send_g(self):
        """The g values to send left and right
        """

        # This is a mess!!!
        # Make sure to clean it up
        sg = [[None, None], [None, None]]
        
        if self.has_left:
            sg[1][0] = self.g[1][0]  + 2.*self.p*self.x[:,0]
        if self.has_right:
            sg[1][1] = self.g[1][-1] - 2.*self.p*self.x[:,-1]
        if self.has_south:
            sg[0][0] = self.g[0][0]  + 2.*self.p*self.x[0, :]
        if self.has_north:
            sg[0][1] = self.g[0][-1] - 2.*self.p*self.x[-1,:]
            
        return tuple(sg)
        

class RecRegion(Region):
    """Extends Region to handle recursive boundary conditions
    """
    def __init__(self, nt, dims, p=1.):
        Region.__init__(self, nt, dims)
        self.p = p
        # Add buffers for recursive g params
        self.g = []
        for i in range(self.n_dims):
            self.g.append({})
            self.g[i][0]  = self.empty_col(i)
            self.g[i][-1] = self.empty_col(i)
            
    def send_g(self, dim, col):
        """The updated g values to send to neighboring regions
        """
        if   col==0:
            return self.g[dim][0]+2.*self.p*self.cols[dim][0]
        elif col==-1:
            return self.g[dim][-1]-2.*self.p*self.cols[dim][-1]
        else:
            return None        
