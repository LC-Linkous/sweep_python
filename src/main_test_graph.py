#! /usr/bin/python3

##--------------------------------------------------------------------\
#   sweep
#   './sweep/src/main_test_graph.py'
#   Test function/example for using the 'sweep' class in sweep.py.
#   Format updates are for integration in the AntennaCAT GUI.
#   This version builds from 'test_details.py' to include a 
#       matplotlib plot of particle location
#
#   Author(s): Lauren Linkous, Jonathan Lundquist
#   Last update: June 3, 2024
##--------------------------------------------------------------------\


import numpy as np
import time
import matplotlib.pyplot as plt
from sweep import sweep
import configs_F as func_configs



class TestGraph():
    def __init__(self):
        # Constant variables
        NO_OF_PARTICLES = 1              # Number of indpendent agents searching the space
        LB = func_configs.LB             # Lower boundaries
        UB = func_configs.UB             # Upper boundaries
        OUT_VARS = func_configs.OUT_VARS # Number of output variables (y-values)
        TARGETS = func_configs.TARGETS   # Target values for output

        MIN_RES = [[0.1, 0.2, 0.3]]      # Minimum resolution for search
        MAX_RES = [[0.01, 0.02, 0.01]]   # Maximum resolution for search
        E_TOL = 10 ** -3                 # Convergence Tolerance. For Sweep, this should be a larger value
        MAXIT = 5000                     # Maximum allowed iterations
        SEARCH_METHOD = 1                # int search 1 = basic_grid, 2 = ...
        
        # Objective function dependent variables
        func_F = func_configs.OBJECTIVE_FUNC  # objective function
        constr_F = func_configs.CONSTR_FUNC   # constraint function


        self.best_eval = 1

        parent = None           # Optional parent class for swarm 
                                        # (Used for passing debug messages or
                                        # other information that will appear 
                                        # in GUI panels)

        self.suppress_output = False   # Suppress the console output for updates

        self.allow_update = True      # Allow objective call to update state 
                                      # (can be set on each iteration)

        detailedWarnings = False      # Optional boolean for detailed feedback


        self.mySweep = sweep(NO_OF_PARTICLES,LB, UB, MIN_RES, MAX_RES, 
                        OUT_VARS, TARGETS, E_TOL, MAXIT,
                        SEARCH_METHOD, func_F, constr_F, parent, detailedWarnings)  
            


        # Matplotlib setup
        self.targets = TARGETS
        self.fig = plt.figure(figsize=(14, 7))
        # position
        self.ax1 = self.fig.add_subplot(121, projection='3d')
        self.ax1.set_title("Particle Location")
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z')
        self.scatter1 = None
        # fitness
        self.ax2 = self.fig.add_subplot(122, projection='3d')
        self.ax2.set_title("Fitness Relation to Target")
        self.ax2.set_xlabel('X')
        self.ax2.set_ylabel('Y')
        self.ax2.set_zlabel('Z')
        self.scatter2 = None

    def debug_message_printout(self, txt):
        if txt is None:
            return
        # sets the string as it gets it
        curTime = time.strftime("%H:%M:%S", time.localtime())
        msg = "[" + str(curTime) +"] " + str(txt)
        print(msg)


    def record_params(self):
        # this function is called from particle_swarm.py to trigger a write to a log file
        # running in the AntennaCAT GUI to record the parameter iteration that caused an error
        pass
         

    def update_plot(self, m_coords, f_coords, targets, showTarget, clearAx=True, setLimts=False):
        # if self.scatter is None:
        if clearAx == True:
            self.ax1.clear() #use this to git rid of the 'ant tunnel' trails
            self.ax2.clear()
        if setLimts == True:
            self.ax1.set_xlim(-5, 5)
            self.ax1.set_ylim(-5, 5)
            self.ax1.set_zlim(-5, 5)
            
            self.ax2.set_xlim(-5, 5)
            self.ax2.set_ylim(-5, 5)
            self.ax2.set_zlim(-5, 5)
        
        # MOVEMENT PLOT
        if np.shape(m_coords)[0] == 2: #2-dim func
            self.ax1.set_title("Particle/Agent Location")
            self.ax1.set_xlabel("$x_1$")
            self.ax1.set_ylabel("$x_2$")
            self.scatter = self.ax1.scatter(m_coords[0, :], m_coords[1, :], edgecolors='b')

        elif np.shape(m_coords)[0] == 3: #3-dim func
            self.ax1.set_title("Particle/Agent Location")
            self.ax1.set_xlabel("$x_1$")
            self.ax1.set_ylabel("$x_2$")
            self.ax1.set_zlabel("$x_3$")
            self.scatter = self.ax1.scatter(m_coords[0, :], m_coords[1, :], m_coords[2, :], edgecolors='b')


        # FITNESS PLOT
        if np.shape(f_coords)[0] == 2: #2-dim obj func
            self.ax2.set_title("Global Best Fitness Relation to Target")
            self.ax2.set_xlabel("$F_{1}(x,y)$")
            self.ax2.set_ylabel("$F_{2}(x,y)$")
            self.scatter = self.ax2.scatter(f_coords[0, :], f_coords[1, :], marker='o', s=40, facecolor="none", edgecolors="k")

        elif np.shape(f_coords)[0] == 3: #3-dim obj fun
            self.ax2.set_title("Global Best Fitness Relation to Target")
            self.ax2.set_xlabel("$F_{1}(x,y)$")
            self.ax2.set_ylabel("$F_{2}(x,y)$")
            self.ax2.set_zlabel("$F_{3}(x,y)$")
            self.scatter = self.ax2.scatter(f_coords[0, :], f_coords[1, :], f_coords[2, :], marker='o', s=40, facecolor="none", edgecolors="k")


        if showTarget == True: # plot the target point
            if len(targets) == 1:
                self.scatter = self.ax2.scatter(targets[0], 0, marker='*', edgecolors='r')
            if len(targets) == 2:
                self.scatter = self.ax2.scatter(targets[0], targets[1], marker='*', edgecolors='r')
            elif len(targets) == 3:
                self.scatter = self.ax2.scatter(targets[0], targets[1], targets[2], marker='*', edgecolors='r')


        plt.pause(0.0001)  # Pause to update the plot




    def run(self):
        

        # instantiation of particle swarm optimizer 
        while not self.mySweep.complete():

            # step through optimizer processing
            self.mySweep.step(self.suppress_output)

            # call the objective function, control 
            # when it is allowed to update and return 
            # control to optimizer
            self.mySweep.call_objective(self.allow_update)
            iter, eval = self.mySweep.get_convergence_data()
            if (eval < self.best_eval) and (eval != 0):
                self.best_eval = eval
            if self.suppress_output:
                if iter%100 ==0: #print out every 100th iteration update
                    print("Iteration")
                    print(iter)
                    print("Best Eval")
                    print(self.best_eval)
            m_coords = self.mySweep.M  #get x,y,z coordinate locations
            f_coords = self.mySweep.F_Gb # global best of set
            self.update_plot(m_coords, f_coords, self.targets, showTarget=True, clearAx=False) #update matplot

        print("Optimized Solution")
        print(self.mySweep.get_optimized_soln())
        print("Optimized Outputs")
        print(self.mySweep.get_optimized_outs())


if __name__ == "__main__":
    tg = TestGraph()
    tg.run()
