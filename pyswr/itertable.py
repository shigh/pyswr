import numpy as np

class IterTable(object):
    """Handle the tricky wrap around logic and adjacent ranks
    
    Put in its own class to keep the waveform code clean
    This should work for all dimensions
    
    Call advance AFTER a full iteration has completed
    
    Time variables should be interperted as solving for time t
    """
    
    def __init__(self, cart, nt, max_itr):
        """Pass in mpi4py cartcomm object
        """
        self.cart    = cart
        self.nt      = nt
        self.max_itr = max_itr
        
        self.cart_dims, _, self.location = self.cart.Get_topo()
        self.n_itr     = self.cart_dims[0]
        self.grid_size = self.cart_dims[1:]
        self.itr_level = self.location[0]
        self.is_first  = self.itr_level == 0
        self.is_last   = self.itr_level == self.n_itr-1
        self.itr       = self.itr_level
        self.rank      = self.cart.Get_rank()
        
        self.t        = 1 if self.is_first else 0
        self.t_next   = 1 if self.is_last  else 0
        self.t_prev   = 1 #if self.itr_level>0 else 0
        self.itr_next = self.itr_level+1 \
                        if self.itr_level<self.n_itr-1 else 0
        self.itr_prev = self.itr_level-1 \
                        if self.itr_level>0 else self.n_itr-1
        self.last_itr = -1
            
    
    @property
    def prev_has_started(self):
        """Has prev started its first iteration?
        """
        if self.is_first: 
            return self.itr>0 or self.t>self.n_itr-1
        else: 
            return self.t_prev>0
    
    @property
    def next_has_started(self):
        """Has next started its first iteration?
        """
        if self.is_last: 
            return True
        else:            
            return self.t>0
    
    @property
    def next_has_finished(self):
        """Has next completed its final iteration?
        """
        return self.itr_next >= self.max_itr
    
    @property
    def prev_has_finished(self):
        """Has prev completed its final iteration?
        """
        return self.itr_prev >= self.max_itr
    
    @property
    def has_finished(self):
        """Has this region completed its final iteration?
        """
        return self.itr >= self.max_itr
    
    # Active properties tell you if the next and 
    # previous iterations have started and have
    # not yet completed
    @property
    def prev_active(self):
        """Is prev still working?
        """
        return self.prev_has_started and \
              (not self.prev_has_finished)
    
    @property
    def next_active(self):
        """Is next still working?
        """
        return self.next_has_started and \
              (not self.next_has_finished)
    
    @property
    def reset_solver(self):
        """Should you reset the solvers init vals?
        """
        return self.t==1
    
    def advance(self):
        """Advances all variables
        """

        if self.next_active:
            self.t_next += 1
            if self.t_next == self.nt: 
                self.t_next = 1
                self.itr_next += self.n_itr

        if self.prev_active:
            self.t_prev += 1
            if self.t_prev == self.nt: 
                self.t_prev = 1
                self.itr_prev += self.n_itr
            
        self.t += 1
        if self.t == self.nt:
            self.t = 1
            self.last_itr = self.itr
            self.itr += self.n_itr
