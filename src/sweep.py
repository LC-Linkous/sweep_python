#! /usr/bin/python3

##--------------------------------------------------------------------\
#   sweep_python
#   './sweep_python/src/sweep.py'
#   Parameter sweep class. Iterates through a specified parameter space
#   to find the optimial solution based on target values.
#
#   Author(s): Lauren Linkous, Jonathan Lundquist
#   Last update: June 26, 2024
##--------------------------------------------------------------------\


import numpy as np
from numpy.random import Generator, MT19937
import sys
import time
np.seterr(all='raise')


class sweep:
    # arguments should take form: 
    # sweep(int, [[float, float,...]], [[float, float,...]], 
    #       [[float, float,...]], 
    #        int, [[float, float,...]],
    #        float, int, int,
    #        func, func, obj, bool) 
    # int search_method:  1 = basic_grid, 2 = random_search
    
    def __init__(self, NO_OF_PARTICLES, lbound, ubound, 
                 step_res,
                 output_size, targets,
                 E_TOL, maxit, search_method, 
                 obj_func, constr_func, parent=None, detailedWarnings=False):  
        
        # Optional parent class func call to write out values that trigger constraint issues
        self.parent = parent 
        # Additional output for advanced debugging to TERMINAL. 
        # Some of these messages will be returned via debugTigger
        self.detailedWarnings = detailedWarnings 

        # problem height and width
        heightl = np.shape(lbound)[0]
        widthl = np.shape(lbound)[1]
        heightu = np.shape(ubound)[0]
        widthu = np.shape(ubound)[1]

        # extract from array
        lbound = np.array(lbound[0])
        ubound = np.array(ubound[0])

        self.rng = Generator(MT19937())

        if ((heightl > 1) and (widthl > 1)) \
           or ((heightu > 1) and (widthu > 1)) \
           or (heightu != heightl) \
           or (widthl != widthu):
            
            if self.parent == None:
                pass
            else:
                self.parent.record_params()
                self.parent.debug_message_printout("ERROR: lbound and ubound must be 1xN-dimensional \
                                                        arrays  with the same length")
           
        else:
            
            
            if heightl == 1:
                lbound = lbound
        
            if heightu == 1:
                ubound = ubound

            self.lbound = lbound.astype(np.float64) 
            self.ubound = ubound.astype(np.float64) 
            variation = ubound-lbound


            # CHECK RESOLUTION ARR VS LOC ARR
            self.search_resolution = np.array(step_res)[0] #default
            if len(self.search_resolution) != len(self.lbound):
                print("ERROR in initialization. resolution array must be the same length as the problem bounds")
                print("RESOLUTION: " + str(self.search_resolution))
                print("PROBLEM BOUNDS LENGTH: " + str(len(self.ubound)))
            else:
                # use NO_OF_PARTICLES to set if this a multi agent search or not
                # first 'particle'
                self.M = np.array([self.lbound])  # at least 1 particle strats at the lower bounds


                # any other agents (if they exist they're assigned random starting locs)
                for i in range(2,int(NO_OF_PARTICLES)+1):
                    new_M = np.multiply((self.rng.random((1,np.max([heightl, widthl])))), variation)+ lbound
                    self.M = np.vstack([self.M, new_M])

                '''
                self.M                      : An array of current search location(s).
                self.output_size            : An integer value for the output size of obj func
                self.Active                 : An array indicating the activity status of each particle.
                self.Gb                     : Global best position, initialized with a large value.
                self.F_Gb                   : Fitness value corresponding to the global best position.
                self.targets                : Target values for the optimization process.
                self.search_resolution      : Step search resolutions.      
                self.current_step           : Current step increment.
                self.maxit                  : Maximum number of iterations.
                self.E_TOL                  : Error tolerance.
                self.obj_func               : Objective function to be optimized.      
                self.constr_func            : Constraint function.  
                self.iter                   : Current iteration count.
                self.current_particle       : Index of the current particle being evaluated.
                self.number_of_particles    : Total number of particles. 
                self.allow_update           : Flag indicating whether to allow updates.
                self.search_method          : search method for the optimization problem.
                self.Flist                  : List to store fitness values.
                self.Fvals                  : List to store fitness values.
                self.Mlast                  : Last search location
                '''
                self.output_size = output_size
                self.Active = np.ones((NO_OF_PARTICLES))  # not/active if particles finish before others
                self.Gb = sys.maxsize*np.ones((1,np.max([heightl, widthl])))   
                self.F_Gb = sys.maxsize*np.ones((1,output_size))              
                self.targets = np.array(targets)
                self.maxit = maxit                       
                self.E_TOL = E_TOL                                              
                self.obj_func = obj_func                                             
                self.constr_func = constr_func                                   
                self.iter = 0    
                self.current_particle = 0     
                self.number_of_particles = NO_OF_PARTICLES                      
                self.allow_update = 0                                           
                self.search_method = search_method                                       
                self.Flist = []                                                 
                self.Fvals = []                                                 
                self.Mlast = 1*self.ubound
                self.no_of_particles = NO_OF_PARTICLES

                self.error_message_generator("sweep successfully initialized")
            

    def error_message_generator(self, msg):
        # for error messages, but also repurposed for general updates
        if self.parent == None:
            print(msg)
        else:
            self.parent.debug_message_printout(msg)
            
    def call_objective(self, allow_update):

        if self.Active[self.current_particle]:
            # call the objective function. If there's an issue with the function execution, 'noError' returns False
            newFVals, noError = self.obj_func(np.hstack(self.M[self.current_particle]), self.output_size)
            if noError == True:
                self.Fvals = [newFVals]
                if allow_update:
                    self.Flist = np.hstack(abs(self.targets - self.Fvals))
                    self.iter = self.iter + 1
                    self.allow_update = 1
                else:
                    self.allow_update = 0
            return noError# return is for error reporting purposes only
        
    
    def check_move_validity(self, particle):
        # make sure that the next move is not outside of the lbounds and ubounds.
        valid_move = True
        for i in range(0,np.shape(self.M)[0]):
            if (self.lbound[i] > self.M[particle,i]) \
               or (self.ubound[i] < self.M[particle,i]):
                    valid_move = False
    
        return valid_move

    def check_bounds(self, particle):
        update = 0
        for i in range(0,(np.shape(self.M)[0])):
            if (self.lbound[i] > self.M[particle,i]) \
               or (self.ubound[i] < self.M[particle,i]):
                update = i+1        
        return update
    
           
    def grid_search(self, particle):
        # If particle is out of bounds, bring the particle back in bounds
        # The first condition checks if constraints are met, 
        # and the second determines if the values are to large (positive or negitive)
        # and may cause a buffer overflow with large exponents (a bug that was found experimentally)
        
        # Convert to numpy arrays for easier manipulation
        lbounds = self.lbound.flatten()
        ubounds = self.ubound.flatten()
        resolution = self.search_resolution

        #check current location to determine which dimension is getting incremented next
        N = len(np.array(self.M[particle]))
        temp_current_loc = 1.0*np.array(self.M[particle]) #
        for i in range(N, 0, -1):
            next_pos = temp_current_loc[i-1] + resolution[i-1]
            if next_pos < lbounds[i-1]: # lower than the lower bounds. 
                # if this is out of bounds, the next iter in the for loop will catch it
                # set current dim to the upper bounds to do a wrap around.
                temp_current_loc[i-1] = ubounds[i-1]
                if (i-1) > 0:
                    pass
                    # the next iteration of the for loop will increment the next dimension
                else:
                    # done. hit the lower bounds
                    self.Active[particle] = 0
            elif next_pos > ubounds[i-1]:
                # if this is out of bounds, the next iter in the for loop will catch it
                # set current dim to the lower bounds to do a wrap around.
                temp_current_loc[i-1] = lbounds[i-1]
                if (i-1) > 0:
                    pass
                    # if this is out of bounds, the next iter in the for loop will catch it
                else:
                    # done
                    self.Active[particle] = 0
            else:
                temp_current_loc[i-1] = next_pos
                break # incremented in 1 dimension at a time
        # set current location
        self.M[particle] = 1.0*temp_current_loc


    def random_search(self, particle):
        # Randomly generate a particle between the upper and lower bounds
        lbounds = self.lbound.flatten()
        ubounds = self.ubound.flatten()

        random_numbers = []

        for lower, upper in zip(lbounds, ubounds):
            rand_num = self.rng.uniform(lower, upper)
            random_numbers.append(rand_num)

        self.M[particle] = random_numbers


    def check_global_local(self, Flist, particle):
        # use L2 norm to check if fitness val is less than global best fitness
        # if yes, update with the new best point
        if np.linalg.norm(Flist) < np.linalg.norm(self.F_Gb):
            self.F_Gb = np.array([Flist])
            self.Gb = np.array(self.M[particle])

    def update_point(self, particle):
        # update the location.
        self.Mlast = 1*self.M
        if self.search_method == 1:
            # grid search
            self.grid_search(particle)
        elif self.search_method == 2:
            # random search
            self.random_search(particle)
        else:
            self.error_message_generator("ERROR: invalid search method selected!")



    def converged(self):
        # check if converged
        convergence = np.linalg.norm(self.F_Gb) < self.E_TOL
        return convergence

    def maxed(self):
        # check if search max iterations hit
        max_iter = self.iter > self.maxit
        return max_iter

    def active_agents(self):
        # check if the search area has been covered.
        active = 0
        for i in self.Active:
            if (i == 1):
                active = active + 1        
        if active > 0:
            return True
        return False
    
    def count_active_agents(self):
        active = 0
        for i in self.Active:
            if (i == 1):
                active = active + 1        
        return active    

    def complete(self):
        done = self.converged() or self.maxed() or (not self.active_agents())
        return done
    
    def step(self, suppress_output):
        if not suppress_output:
            msg = "\n-----------------------------\n" + \
                "STEP #" + str(self.iter) +"\n" + \
                "-----------------------------\n" + \
                "Current Location:\n" + \
                str(self.M[self.current_particle]) +"\n" + \
                "Best Fitness Solution: \n" +\
                str(np.linalg.norm(self.F_Gb)) +"\n" +\
                "Best Particle Position: \n" +\
                str(np.hstack(self.Gb)) + "\n" +\
                "Current Search Resolution\n" + \
                str(self.search_resolution) +"\n" + \
                "Number of Active Particles\n" + \
                str(self.count_active_agents()) +"\n" + \
                "-----------------------------"
            self.error_message_generator(msg)
           
        if self.allow_update:
            if self.Active[self.current_particle]:
                # check if points are better than last global bests
                self.check_global_local(self.Flist,self.current_particle)
                # move point(s) based on movement model
                self.update_point(self.current_particle)
            # if we've iterated through all of the points, start at begining 
            self.current_particle = self.current_particle + 1
            if self.current_particle == self.number_of_particles:
                self.current_particle = 0
            if self.complete() and not suppress_output:
                msg =  "\nPoints: \n" + str(self.Gb) + "\n" + \
                    "Iterations: \n" + str(self.iter) + "\n" + \
                    "Flist: \n" + str(self.F_Gb) + "\n" + \
                    "Norm Flist: \n" + str(np.linalg.norm(self.F_Gb)) + "\n"
                self.error_message_generator(msg)

    def get_obj_inputs(self):
        return np.vstack(self.M)
    
    def get_convergence_data(self):
        best_eval = np.linalg.norm(self.F_Gb)
        iteration = 1*self.iter
        return iteration, best_eval
        
    def get_optimized_soln(self):
        return self.Gb 
    
    def get_optimized_outs(self):
        return self.F_Gb
    
    def absolute_mean_deviation_from_target(self):
        mean_data = np.vstack(np.mean(self.M,axis=1))
        abs_data = np.zeros(np.shape(self.M))
        for i in range(0,self.number_of_particles):
            abs_data[:,i] = np.squeeze(np.abs(np.vstack(self.M[:,i])-mean_data))
        abs_mean_dev = np.linalg.norm(np.mean(abs_data,axis=1))
        return abs_mean_dev

