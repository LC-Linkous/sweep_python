#! /usr/bin/python3

##--------------------------------------------------------------------\
#   sweep
#   './sweep/src/test.py'
#   Test function/example for using the 'sweep' class in sweep.py.
#   Format updates are for integration in the AntennaCAT GUI.
#
#   Author(s): Jonathan Lundquist, Lauren Linkous
#   Last update: May 31, 2024
##--------------------------------------------------------------------\


import numpy as np
from sweep import sweep
from func_F import func_F
from constr_F import constr_F


if __name__ == "__main__":
    NO_OF_PARTICLES = 1             # Number of indpendent agents searching the space
    LB = [[0.21, 0, 0.1]]           # Lower boundaries
    UB = [[1, 1, 0.5]]              # Upper boundaries
    MIN_RES = [[0.01, 0.02, 0.01]]  # Minimum resolution for search
    MAX_RES = [[0.01, 0.02, 0.01]]  # Maximum resolution for search
    OUT_VARS = 2                    # Number of output variables (y-values)
    TARGETS = [0, 0]                # Target values for output
    E_TOL = 10 ** -6                # Convergence Tolerance
    MAXIT = 200                     # Maximum allowed iterations
    SEARCH_METHOD = 1               # int search 1 = basic_grid, 2 = ...
    best_eval = 1

    parent = None            # Optional parent class for swarm 
                                        # (Used for passing debug messages or
                                        # other information that will appear 
                                        # in GUI panels)

    suppress_output = False   # Suppress the console output of particle swarm

    allow_update = True       # Allow objective call to update state 
                                # (can be set on each iteration)

    mySweep = sweep(NO_OF_PARTICLES, LB, UB, MIN_RES, MAX_RES, 
                    OUT_VARS, TARGETS, E_TOL, MAXIT,
                    SEARCH_METHOD, func_F, constr_F)  

    # instantiation of sweep optimizer 
    while not mySweep.complete():

        # step through optimizer processing
        # update_search, will change the particle location
        mySweep.step(suppress_output)

        # call the objective function, control 
        # when it is allowed to update and return 
        # control to optimizer

        # for some objective functions, the function
        # might not evaluate correctly (e.g., under/overflow)
        # so when that happens, the function is not evaluated
        # and the 'step' fucntion will re-gen values and try again

        mySweep.call_objective(allow_update)
        iter, eval = mySweep.get_convergence_data()
        if (eval < best_eval) and (eval != 0):
            best_eval = eval
        if suppress_output:
            if iter%100 ==0: #print out every 100th iteration update
                print("Iteration")
                print(iter)
                print("Best Eval")
                print(best_eval)

    print("Optimized Solution")
    print(mySweep.get_optimized_soln())
    print("Optimized Outputs")
    print(mySweep.get_optimized_outs())
