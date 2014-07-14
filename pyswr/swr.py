
from pyswr.itertable import *
from pyswr.utils import *


def swr_1d_heat(MPI, comm, dims, region, solver, f0, steps, offset):

    rank = comm.rank
    size = comm.size

    has_left  = rank>0
    has_right = rank<size-1

    right = rank+1
    left  = rank-1

    if has_right:
        region.add_col(0, -1-offset)
    if has_left:
        region.add_col(0, offset)


    nt = region.nt
    for _ in range(steps):
        solver.x[:] = f0

        for i in range(1, nt):
            solver.left  = region.cols[0][0][i]
            solver.right = region.cols[0][-1][i]
            
            solver.solve()
            region.update_cols(i, solver.x)

        send_requests = []
        if has_right:
            rr = comm.Isend(region.cols[0][-1-offset], dest=right)
            send_requests.append(rr)
        if has_left:
            rl = comm.Isend(region.cols[0][offset], dest=left)
            send_requests.append(rl)

        if has_right:
            comm.Recv(region.cols[0][-1], source=right)
        if has_left:
            comm.Recv(region.cols[0][0], source=left)

        MPI.Request.Waitall(send_requests)


def swr_opt_heat(MPI, comm, dims, region, solver, steps):

    dims = as_tuple(dims)
    n_dims = len(dims)
    
    rank   = comm.rank
    size   = comm.size
    cart   = comm.Create_cart(dims)
    coords = cart.Get_coords(rank)
    
    nb = []
    for dim in range(n_dims):
        nb.append([-1, -1])
        loc  = coords[dim]
        dmax = dims[dim]
        if loc>0:
            left_loc = list(coords)
            left_loc[dim] -= 1
            left = cart.Get_cart_rank(tuple(left_loc))
            nb[-1][0]  = left
            
        if loc<dmax-1:
            right_loc = list(coords)
            right_loc[dim] += 1
            right = cart.Get_cart_rank(tuple(right_loc))
            nb[-1][-1]  = right

    nt = region.nt
    for step in range(steps):

        # Reset solver for next iteration
        solver.x[:] = region.slices[0]

        # Apply solver over each time step
        for t in range(1, nt):
            solver.g = [[region.g[dim][i][t] for i in [0,-1]]
                        for dim in range(n_dims)]

            solver.solve()
            region.update_cols(t, solver.x)

        send_requests = []
        for dim in range(n_dims):
            if nb[dim][0]!=-1:
                rr = comm.Isend(region.send_g(dim, 0),  dest=nb[dim][0])
                send_requests.append(rr)
            if nb[dim][-1]!=-1:
                rr = comm.Isend(region.send_g(dim, -1), dest=nb[dim][-1])
                send_requests.append(rr)

        for dim in range(n_dims):
            if nb[dim][0]!=-1:
                comm.Recv(region.g[dim][0],  source=nb[dim][0])
            if nb[dim][-1]!=-1:
                comm.Recv(region.g[dim][-1], source=nb[dim][-1])


        MPI.Request.Waitall(send_requests)
    

def swr_1dopt_pipe_heat(MPI, comm, dims, region, solver, steps):

    dims   = as_tuple(dims)
    n_dims = len(dims)-1

    rank = comm.rank
    size = comm.size
    periods    = [False]*(n_dims+1)
    periods[0] = True
    cart   = comm.Create_cart(dims, periods=periods)
    coords = cart.Get_coords(rank)
    
    nb = []
    for dim in range(1, n_dims+1):
        nb.append([[-1, -1], [-1, -1]])
        loc  = coords[dim]
        dmax = dims[dim]
        if loc>0:
            left_loc = list(coords)
            left_loc[dim] -= 1
            # Prev itr
            left_loc[0]    = coords[0]-1
            left = cart.Get_cart_rank(tuple(left_loc))
            nb[-1][0][0]   = left
            # Next itr
            left_loc[0]    = coords[0]+1
            left = cart.Get_cart_rank(tuple(left_loc))
            nb[-1][+1][0]  = left
            
        if loc<dmax-1:
            right_loc = list(coords)
            right_loc[dim] += 1
            # Prev itr
            right_loc[0]    = coords[0]-1
            right = cart.Get_cart_rank(tuple(right_loc))
            nb[-1][0][-1]   = right
            # Next itr
            right_loc[0]    = coords[0]+1
            right = cart.Get_cart_rank(tuple(right_loc))
            nb[-1][+1][-1]  = right

    nt   = region.nt
    it_table = IterTable(cart, nt, steps)
    while not it_table.has_finished:

        t = it_table.t
        # Reset for next iteration
        if it_table.reset_solver:
            solver.x[:] = region.slices[0]

        if t>0:
            solver.g = [[region.g[dim][i][t] for i in [0,-1]]
                        for dim in range(n_dims)]
            solver.solve()
            region.update_cols(t, solver.x)

        send_requests = []            
        if it_table.next_active and t>0:

            for dim in range(n_dims):
                if nb[dim][1][0]!=-1:
                    rr = comm.Isend(region.send_g(dim, 0)[t:t+1],  dest=nb[dim][1][0])
                    send_requests.append(rr)
                if nb[dim][1][-1]!=-1:
                    rr = comm.Isend(region.send_g(dim, -1)[t:t+1], dest=nb[dim][1][-1])
                    send_requests.append(rr)

        if it_table.prev_active:

            tp = it_table.t_prev
            for dim in range(n_dims):
                if nb[dim][0][0]!=-1:
                    comm.Recv(region.g[dim][0][tp:tp+1],  source=nb[dim][0][0])
                if nb[dim][0][-1]!=-1:
                    comm.Recv(region.g[dim][-1][tp:tp+1], source=nb[dim][0][-1])

        MPI.Request.Waitall(send_requests)

        it_table.advance()

    return it_table

    
def swr_2dopt_pipe_heat(MPI, comm, dims, region, solver, steps):

    nt = region.nt

    rank  = comm.rank
    size  = comm.size

    ########################################
    # Set up MPI cart and location variables

    n_itr, n_reg_y, n_reg_x = dims

    cart   = comm.Create_cart((n_itr, n_reg_y, n_reg_x))
    # The location of this nodes region
    itr, ry, rx = cart.Get_coords(rank)
    it_table = IterTable(cart, nt, steps)

    # Determine which adjacent nodes exist
    has_left  = rx>0
    has_right = rx<n_reg_x-1
    has_north = ry<n_reg_y-1
    has_south = ry>0
    has_prev_itr = itr>0       or itr==0
    has_next_itr = itr<n_itr-1 or itr==n_itr-1
    

    left = right = north = south    = None
    next_itr_left  = next_itr_right = None
    next_itr_north = next_itr_south = None
    prev_itr_left  = prev_itr_right = None
    prev_itr_north = prev_itr_south = None

    if itr!=0 and itr!=n_itr-1:
        if has_left:
            prev_itr_left  = cart.Get_cart_rank((itr-1, ry, rx-1))
            next_itr_left  = cart.Get_cart_rank((itr+1, ry, rx-1))
        if has_right:
            prev_itr_right = cart.Get_cart_rank((itr-1, ry, rx+1))
            next_itr_right = cart.Get_cart_rank((itr+1, ry, rx+1))
        if has_north:
            prev_itr_north = cart.Get_cart_rank((itr-1, ry+1, rx))
            next_itr_north = cart.Get_cart_rank((itr+1, ry+1, rx))
        if has_south:
            prev_itr_south = cart.Get_cart_rank((itr-1, ry-1, rx))
            next_itr_south = cart.Get_cart_rank((itr+1, ry-1, rx))
    elif itr==0:
        if has_left:
            prev_itr_left  = cart.Get_cart_rank((n_itr-1, ry, rx-1))
            next_itr_left  = cart.Get_cart_rank((itr+1, ry, rx-1))
        if has_right:
            prev_itr_right = cart.Get_cart_rank((n_itr-1, ry, rx+1))
            next_itr_right = cart.Get_cart_rank((itr+1, ry, rx+1))
        if has_north:
            prev_itr_north = cart.Get_cart_rank((n_itr-1, ry+1, rx))
            next_itr_north = cart.Get_cart_rank((itr+1, ry+1, rx))
        if has_south:
            prev_itr_south = cart.Get_cart_rank((n_itr-1, ry-1, rx))
            next_itr_south = cart.Get_cart_rank((itr+1, ry-1, rx))                
    elif itr==n_itr-1:
        if has_left:
            next_itr_left  = cart.Get_cart_rank((0, ry, rx-1))
            prev_itr_left  = cart.Get_cart_rank((itr-1, ry, rx-1))
        if has_right:
            next_itr_right = cart.Get_cart_rank((0, ry, rx+1))
            prev_itr_right = cart.Get_cart_rank((itr-1, ry, rx+1))
        if has_north:
            next_itr_north = cart.Get_cart_rank((0, ry+1, rx))
            prev_itr_north = cart.Get_cart_rank((itr-1, ry+1, rx))
        if has_south:
            next_itr_south = cart.Get_cart_rank((0, ry-1, rx))
            prev_itr_south = cart.Get_cart_rank((itr-1, ry-1, rx))
    

    # Apply solver over each time step
    while not it_table.has_finished:

        t = it_table.t

        # Reset for next iteration
        if it_table.reset_solver:
            solver.x[:] = region.slices[0]

        send_requests = []        

        if t>0:
            solver.g = [[region.g[0][0][t], region.g[0][-1][t]],
                        [region.g[1][0][t], region.g[1][-1][t]]]
            solver.solve()
            region.update_cols(t, solver.x)
            #update_last_slice_vals()

        # Communicate with adjacent regions
        if it_table.next_active and t>0:
            if has_right:
                rnr = comm.Isend(region.send_g(1, -1)[t:t+1], dest=next_itr_right)
                send_requests.append(rnr)
            if has_left:
                rnl = comm.Isend(region.send_g(1, 0)[t:t+1], dest=next_itr_left)
                send_requests.append(rnl)
            if has_north:
                rnn = comm.Isend(region.send_g(0, -1)[t:t+1], dest=next_itr_north)
                send_requests.append(rnn)
            if has_south:
                rns = comm.Isend(region.send_g(0, 0)[t:t+1], dest=next_itr_south)
                send_requests.append(rns)

        if it_table.prev_active:
            tp = it_table.t_prev
            if has_right:
                comm.Recv(region.g[1][-1][tp:tp+1], source=prev_itr_right)
            if has_left:
                comm.Recv(region.g[1][0][tp:tp+1], source=prev_itr_left)
            if has_north:
                comm.Recv(region.g[0][-1][tp:tp+1], source=prev_itr_north)
            if has_south:
                comm.Recv(region.g[0][0][tp:tp+1], source=prev_itr_south)

        MPI.Request.Waitall(send_requests)

        it_table.advance()

    return it_table
