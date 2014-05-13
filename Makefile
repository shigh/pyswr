

run_1d : 
	@echo With s=1
	@mpirun -n 3 python heatex_1d.py --error -s 1
	@echo With s=10
	@mpirun -n 3 python heatex_1d.py --error -s 10

run_1dopt : 
	@echo With s=1
	@mpirun -n 3 python heatex_1dopt.py --error -s 1
	@echo With s=10
	@mpirun -n 3 python heatex_1dopt.py --error -s 10


run_2dopt : 
	@echo With s=1
	@mpirun -n 6 python heatex_2dopt.py --error -s 1 --reg-x 3 --reg-y 2 -x 50 -y 50
	@echo With s=10
	@mpirun -n 6 python heatex_2dopt.py --error -s 10 --reg-x 3 --reg-y 2 -x 50 -y 50
