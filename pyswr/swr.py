
from pyswr.itertable import *


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


def swr_1dopt_heat(MPI, comm, dims, region, solver, steps):

    rank  = comm.rank
    size  = comm.size
    
    has_left  = rank>0
    has_right = rank<size-1

    right = rank+1
    left  = rank-1

    nt = region.nt
    for step in range(steps):

        # Reset solver for next iteration
        solver.x[:] = region.slices[0]

        # Apply solver over each time step
        for t in range(1, nt):
            solver.g = [region.g[0][0][t], region.g[0][-1][t]]
            solver.solve()
            region.update_cols(t, solver.x)
            # update_last_slice_vals(t, step)

        send_requests = []
        if has_right:
            rr = comm.Isend(region.send_g(0, -1), dest=right)
            send_requests.append(rr)
        if has_left:
            rl = comm.Isend(region.send_g(0, 0), dest=left)
            send_requests.append(rl)

        if has_right:
            comm.Recv(region.g[0][-1], source=right)
        if has_left:
            comm.Recv(region.g[0][0], source=left)

        MPI.Request.Waitall(send_requests)
    

def swr_1dopt_pipe_heat(MPI, comm, dims, region, solver, steps):

    rank  = comm.rank
    size  = comm.size

    nt = region.nt
    n_itr, n_reg = dims
    # TODO Check if comm is a cart
    cart   = comm.Create_cart(dims)
    # The location of this nodes region
    itr, reg = cart.Get_coords(rank)
    it_table = IterTable(cart, nt, steps)

    has_left     = reg>0
    has_right    = reg<n_reg-1
    has_prev_itr = itr>0       or itr==0
    has_next_itr = itr<n_itr-1 or itr==n_itr-1

    right = left = next_itr = prev_itr = None
    next_itr_left = next_itr_right     = None
    prev_itr_left = prev_itr_right     = None


    if itr!=0 and itr!=n_itr-1:
        if has_left:
            prev_itr_left  = cart.Get_cart_rank((itr-1, reg-1))
            next_itr_left  = cart.Get_cart_rank((itr+1, reg-1))
        if has_right:
            prev_itr_right = cart.Get_cart_rank((itr-1, reg+1))
            next_itr_right = cart.Get_cart_rank((itr+1, reg+1))            
    elif itr==0:
        if has_left:
            prev_itr_left  = cart.Get_cart_rank((n_itr-1, reg-1))
            next_itr_left  = cart.Get_cart_rank((itr+1, reg-1))
        if has_right:
            prev_itr_right = cart.Get_cart_rank((n_itr-1, reg+1))
            next_itr_right = cart.Get_cart_rank((itr+1, reg+1))
    elif itr==n_itr-1:
        if has_left:
            next_itr_left  = cart.Get_cart_rank((0, reg-1))
            prev_itr_left  = cart.Get_cart_rank((itr-1, reg-1))
        if has_right:
            next_itr_right = cart.Get_cart_rank((0, reg+1))
            prev_itr_right = cart.Get_cart_rank((itr-1, reg+1))
    

    while not it_table.has_finished:

        t = it_table.t

        # Reset for next iteration
        if it_table.reset_solver:
            solver.x[:] = region.slices[0]

        send_requests = []

        if t>0:
            solver.g = [region.g[0][0][t], region.g[0][-1][t]]
            solver.solve()
            region.update_cols(t, solver.x)

        if it_table.next_active and t>0:
            if has_right:
                rnr = comm.Isend(region.send_g(0, -1)[t:t+1], dest=next_itr_right)
                send_requests.append(rnr)
            if has_left:
                rnl = comm.Isend(region.send_g(0, 0)[t:t+1], dest=next_itr_left)
                send_requests.append(rnl)

        if it_table.prev_active:
            tp = it_table.t_prev
            if has_right:
                comm.Recv(region.g[0][-1][tp:tp+1], source=prev_itr_right)
            if has_left:
                comm.Recv(region.g[0][0][tp:tp+1], source=prev_itr_left)


        MPI.Request.Waitall(send_requests)

        it_table.advance()

    return it_table
