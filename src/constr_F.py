#! /usr/bin/python3

##--------------------------------------------------------------------\
#   sweep
#   './sweep/src/constr_F.py'
#   Function for objective function constraints.
#   Has 2 checks: 1 for the function limitations, 1 for float size
#   Returns True if X array passes constraints check, False otherwise   
#
#   Author(s): Jonathan Lundquist, Lauren Linkous
#   Last update: May 30, 2024
##--------------------------------------------------------------------\


def constr_F(X):
    F = True
    # objective function/problem constraints
    if (X[2] > X[0]/2) or (X[2] < 0.1):
        F = False

    return F