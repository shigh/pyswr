
ALL_RUNS = run_1d run_1dopt run_1dopt_pipe run_2dopt run_2dopt_pipe

all : $(ALL_RUNS)

run_1d :
	@echo
	@echo With s=1
	@mpirun -n 3 python heatex_1d.py --error -s 1
	@echo With s=10
	@mpirun -n 3 python heatex_1d.py --error -s 10

run_1dopt :
	@echo
	@echo With s=1
	@mpirun -n 3 python heatex_1dopt.py --error -s 1
	@echo With s=10
	@mpirun -n 3 python heatex_1dopt.py --error -s 10

run_1dopt_pipe :
	@echo
	@echo With s=1
	@mpirun -n 6 python heatex_1dopt_pipe.py --error -s 1  -r 3
	@echo With s=10
	@mpirun -n 6 python heatex_1dopt_pipe.py --error -s 10 -r 3

run_2dopt :
	@echo
	@echo With s=1
	@mpirun -n 6 python heatex_2dopt.py --error -s 1 --reg-x 3 --reg-y 2 -x 50 -y 50
	@echo With s=10
	@mpirun -n 6 python heatex_2dopt.py --error -s 10 --reg-x 3 --reg-y 2 -x 50 -y 50

run_2dopt_pipe :
	@echo
	@echo With s=1
	@mpirun -n 12 python heatex_2dopt_pipe.py --error -s 1 --reg-x 3 --reg-y 2 -x 50 -y 50
	@echo With s=10
	@mpirun -n 12 python heatex_2dopt_pipe.py --error -s 10 --reg-x 3 --reg-y 2 -x 50 -y 50
