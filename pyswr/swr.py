
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

