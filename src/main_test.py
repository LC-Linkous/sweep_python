#! /usr/bin/python3

##--------------------------------------------------------------------\
#   sweep_python
#   './sweep_python/src/main_test.py'
#   Test function/example for using the 'sweep' class in sweep.py.
#   Format updates are for integration in the AntennaCAT GUI.
#
#   Author(s): Lauren Linkous, Jonathan Lundquist
#   Last update: March 13, 2025
##--------------------------------------------------------------------\


import pandas as pd
from sweep import sweep

# OBJECTIVE FUNCTION SELECTION
#import one_dim_x_test.configs_F as func_configs     # single objective, 1D input
import himmelblau.configs_F as func_configs         # single objective, 2D input
#import lundquist_3_var.configs_F as func_configs     # multi objective function



if __name__ == "__main__":
    NO_OF_PARTICLES = 3             # Number of indpendent agents searching the space
    MIN_RES = [0.1]                 # min resolution for search
    MAX_RES = [1.1]                 # max resolution for search
    TOL = 10 ** -6                  # Convergence Tolerance
    MAXIT = 10000                   # Maximum allowed iterations
    SEARCH_METHOD = 1               # int search 1 = basic_grid, 2 = random_search

    # Objective function dependent variables
    LB = func_configs.LB                    # Lower boundaries, [[0.21, 0, 0.1]]
    UB = func_configs.UB                    # Upper boundaries, [[1, 1, 0.5]]
    IN_VARS = func_configs.IN_VARS          # Number of input variables (x-values)   
    OUT_VARS = func_configs.OUT_VARS        # Number of output variables (y-values)
    TARGETS = func_configs.TARGETS          # Target values for output

    # Objective function dependent variables
    func_F = func_configs.OBJECTIVE_FUNC  # objective function
    constr_F = func_configs.CONSTR_FUNC   # constraint function


    best_eval = 999          # set higher than normal because of the potential for missing the target

    parent = None            # Optional parent class for swarm 
                                        # (Used for passing debug messages or
                                        # other information that will appear 
                                        # in GUI panels)

    suppress_output = True   # Suppress the console output of particle swarm

    allow_update = True       # Allow objective call to update state 
                                # (can be set on each iteration)

    # Constant variables
    opt_params = {'NO_OF_PARTICLES': [NO_OF_PARTICLES],     # Number of indpendent agents searching the space
                'SEARCH_METHOD': [SEARCH_METHOD],           # int search 1 = basic_grid, 2 = random_search
                'MIN_RES': [MIN_RES],                       # min resolution for search
                'MAX_RES': [MAX_RES]}                       # max resolution for search
        

    opt_df = pd.DataFrame(opt_params)
    mySweep = sweep(LB, UB, TARGETS, TOL, MAXIT,
                            func_F, constr_F,
                            opt_df,
                            parent=parent)   

    # instantiation of sweep optimizer 
    while not mySweep.complete():

        # step through optimizer processing
        # update_search, will change the particle location
        mySweep.step(suppress_output)

        # call the objective function, control 
        # when it is allowed to update and return 
        # control to optimizer

        mySweep.call_objective(allow_update)
        iter, eval = mySweep.get_convergence_data()
        if (eval < best_eval) and (eval != 0):
            best_eval = eval
        if suppress_output:
            if iter%10 ==0: #print out every 100th iteration update
                print("Iteration")
                print(iter)
                print("Best Eval")
                print(best_eval)

    print("Optimized Solution")
    print(mySweep.get_optimized_soln())
    print("Optimized Outputs")
    print(mySweep.get_optimized_outs())

