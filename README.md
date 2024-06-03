# sweep

Simple sweep optimizer written in Python. Currently uses a basic grid search

The format is based off of the [adaptive timestep PSO optimizer](https://github.com/jonathan46000/pso_python) by [jonathan46000](https://github.com/jonathan46000) for data collection baseline. This repo does not feature any PSO optimization. Instead, the format has been used to retain modularity with other optimizers.

Now featuring AntennaCAT hooks for GUI integration and user input handling.

## Table of Contents
* [Requirements](#requirements)
* [Implementation](#implementation)
    * [Constraint Handling](#constraint-handling)
    * [Search Types](#search-types)
        * [basic_grid](#basic_grid)
    * [Multi-Object Optimization](#multi-object-optimization)
    * [Objective Function Handling](#objective-function-handling)
      * [Internal Objective Function Example](internal-objective-function-example)
* [Error Handling](#error-handling)
* [Example Implementations](#example-implementations)
    * [Basic Sweep Example](#basic-sweep-example)
    * [Detailed Messages](#detailed-messages)
    * [Realtime Graph](#realtime-graph)
* [References](#references)
* [Publications and Integration](#publications-and-integration)
* [Licensing](#licensing)  

## Requirements

This project requires numpy and matplotlib. 

Use 'pip install -r requirements.txt' to install the following dependencies:

```python
contourpy==1.2.1
cycler==0.12.1
fonttools==4.51.0
importlib_resources==6.4.0
kiwisolver==1.4.5
matplotlib==3.8.4
numpy==1.26.4
packaging==24.0
pillow==10.3.0
pyparsing==3.1.2
python-dateutil==2.9.0.post0
six==1.16.0
zipp==3.18.1

```

## Implementation
### Constraint Handling
Users must create their own constraint function for their problems, if there are constraints beyond the problem bounds.  This is then passed into the constructor. If the default constraint function is used, it always returns true (which means there are no constraints).

### Search Types
More search types will be added, but for initial deployment, a standard grid search is used.

The variables from PSO_python have been reused to retain modularity with AntennaCAT and provide some consistency between optimizers. 

```
            self.M                      : An array of current search location(s).
            self.output_size            : An integer value for the output size of obj func
            self.Active                 : An array indicating the activity status of each particle.
            self.Gb                     : Global best position, initialized with a large value.
            self.F_Gb                   : Fitness value corresponding to the global best position.
            self.targets                : Target values for the optimization process.
            self.min_search_res         : Minimum search resolution value array.
            self.max_search_res         : Maximum search resolution value array.
            self.search_resolution      : Current search resolutions.      
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
```

```python
        # Constant variables
        NO_OF_PARTICLES = 1              # Number of indpendent agents searching the space
        LB = func_configs.LB             # Lower boundaries
        UB = func_configs.UB             # Upper boundaries
        OUT_VARS = func_configs.OUT_VARS # Number of output variables (y-values)
        TARGETS = func_configs.TARGETS   # Target values for output

        MIN_RES = [[0.1, 0.2, 0.3]]      # Minimum resolution for search
        MAX_RES = [[0.01, 0.02, 0.01]]   # Maximum resolution for search
        E_TOL = 10 ** -3                 # Convergence Tolerance. For Sweep, this should be a larger value
        MAXIT = 5000                     # Maximum allowed iterations (useful for debug)
        SEARCH_METHOD = 1                # int search 1 = basic_grid, 2 = ...

```



#### basic_grid

The basic grid search uses the current position of a particle (or agent), and increments it one step towards the upper bounds based on the
defined problem space. It can use 1 or more particles (or agents) to search a space. If one particle is used, it will start at the lower bound of the decision space, and increment based on the minimum resolution until the particle reaches the maximum boundary limit.

Resolution is a multi-dimensional vector to allow for tuning in all dimensions of the input space. 

This method does not tend to converge with a small error tolerance.


### Multi-Object Optimization
The no preference method of multi-objective optimization, but a Pareto Front is not calculated. Instead the best choice (smallest norm of output vectors) is listed as the output.

### Objective Function Handling
The optimizer minimizes the absolute value of the difference from the target outputs and the evaluated outputs.  Future versions may include options for function minimization absent target values. 

#### Internal Objective Function Example
The current internal optimization function takes 3 inputs, and has 2 outputs. It was created as a simple 3-variable optimization objective function that would be quick to converge.  
<p align="center">
        <img src="https://github.com/LC-Linkous/sweep/blob/main/media/obj_func_pareto.png" alt="Function Feasable Decision Space and Objective Space with Pareto Front" height="200">
</p>
   <p align="center">Function Feasable Decision Space and Objective Space with Pareto Front</p>


```math
\text{minimize}: 
\begin{cases}
f_{1}(\mathbf{x}) = (x_1-0.5)^2 + (x_2-0.1)^2 \\
f_{2}(\mathbf{x}) = (x_3-0.2)^4
\end{cases}
```

| Num. Input Variables| Boundary | Constraints |
|----------|----------|----------|
| 3      | $0.21\leq x_1\leq 1$ <br> $0\leq x_2\leq 1$ <br> $0.1 \leq x_3\leq 0.5$  | $x_3\gt \frac{x_1}{2}$ or $x_3\lt 0.1$| 


This function has three files:
   1) configs_F.py - contains imports for the objective function and constraints, CONSTANT assignments for functions and labeling, boundary ranges, the number of input variables, the number of output values, and the target values for the output
   2) constr_F.py - contains a function with the problem constraints, both for the function and for error handling in the case of under/overflow. 
   3) func_F.py - contains a function with the objective function.

Other multi-objective functions can be applied to this project by following the same format (and several have been collected into a compatable library, and will be realeased in a seperate repo)

## Error Handling

In the sweep.py class, the objective function is called twice. Some optimizer/objective function/parameter combinations cause under/overflows when using numpy. It is a known bug in numpy that as of 5/2024 basic numpy cannot convert floats to longFloats or float128().

 * 1) When the constraints are called to verify if the particle is in bounds, and to apply the selected boundary method. At this point, the 'noErrors' boolean is used to confirm if the objection function resolves. If the objective function does not resolve, or the particle is out of bounds, the boundary conditions are applied.
 * 2) To evaluate the objective function as part of the traditional particle swarm algorithm


## Example Implementations

### Basic Sweep Example
main_test.py provides a sample use case of the optimizer. 

### Detailed Messages
main_test_details.py provides an example using a parent class, and the self.suppress_output and detailedWarnings flags to control error messages that are passed back to the parent class to be printed with a timestamp. This implementation sets up the hooks for integration with AntennaCAT in order to provide the user feedback of warnings and errors.

### Realtime Graph

<p align="center">
        <img src="https://github.com/LC-Linkous/sweep/blob/main/media/sweep_loc_fitness.png" alt="Example Sweep Convergence (Attempt)" height="200">
</p>

main_test_graph.py provides an example using a parent class, and the self.suppress_output and detailedWarnings flags to control error messages that are passed back to the parent class to be printed with a timestamp. Additionally, a realtime graph shows particle locations at every step.

The figure above is a snapshot of the search. The left shows all of the search locations of a single particle (NOTE: toggle a the 'clear' boolean to turn this feature off), and the right side shows the target (marked by a star) and the fitness function locations (the open circles). While the fitness of the particle is very close to the target, it does not come closer than the 10E-6 tolerance, so the search does not converge.

NOTE: if you close the graph as the code is running, the code will continue to run, but the graph will not re-open.


## References
TODO: add in references to papers as search options are added

## Publications and Integration
This software works as a stand-alone implementation, and as one of the optimizers integrated into AntennaCAT.

Publications featuring the code in this repo will be added as they become public.

## Licensing

The code in this repository has been released under GPL-2.0
