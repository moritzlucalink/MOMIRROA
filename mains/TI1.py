#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 16:21:49 2024

@author: moritz
"""

from pyomo.environ import *
from pyomo.core.expr.current import identify_variables, identify_components

import sys

sys.path.append('/home/moritz/coding-morice/PhD_codes/ThesisCodes/methods')

from compute_enclosure import *

"""
problem instance (TI1) from 
Eichfelder, G. and Gerlach, T. and Warnow, L.
Test Instances for Multiobjective  Mixed-Integer Nonlinear Optimization.
2023. 
"""

plt.close('all')

class structure():
    pass

# define the pyomo model
def build_model(m):
    
    model = ConcreteModel()
    
    model.x1 = Var(within=Reals, bounds=(-2,2))
    model.x2 = Var(within=Integers, bounds=(-4,4))

    model.cons0 = Constraint(expr = (model.x1 - 2)**2 + (model.x2 - 2)**2 - 36 <= 0)

    model.objective0 = Objective(expr = model.x1 + model.x2)
    model.objective1 = Objective(expr = model.x1**2 + model.x2**2)
    
    for o in model.component_objects(Objective):
        if not 'objective'+str(m) in o.name:
            o.deactivate()
            
    return model

# set up the parameters
parameter = structure()
parameter.m = 2
parameter.tol = 0.1
parameter.maxiter = 1000
parameter.timeout = 3600
parameter.factor_delta = 0.95 * parameter.tol
parameter.factor_delta_2nd = 0.95 * parameter.tol
parameter.rel_tol = 0.05

# set up options
options = structure()
options.show_plots = True

encl_dict, it = compute_enclosure(build_model, parameter, options)

print('done')