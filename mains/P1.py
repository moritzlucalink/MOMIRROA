#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 12:32:43 2024

@author: moritz
"""

from pyomo.environ import *
from pyomo.core.expr.current import identify_variables, identify_components

import sys

sys.path.append('/home/moritz/coding-morice/PhD_codes/ThesisCodes/MOMIRROA_optimizedmethods')
sys.path.append('/home/moritz/coding-morice/PhD_codes/ThesisCodes/MOMIBB_methods')

from compute_enclosure import *
from MOMIBB_direct import *

"""
problem instance (P1) from
Eichfelder, G. and Stein, O. and Warnow, L.
A deterministic solver for multi-objective mixed-integer convex and nonconvex 
optimization.
.... 2023.
"""

plt.close('all')

class structure():
    pass

# define the pyomo model
def build_model(m):
    
    model = ConcreteModel()
    
    # define variables
    model.x1 = Var(within=Reals, bounds=(0,1))
    model.x2 = Var(within=Reals, bounds=(0,1))
    model.x3 = Var(within=Integers, bounds=(-4,1))
    
    # define constraints
    model.cons0 = Constraint(expr = (1 - model.x1**2 - model.x2**2) <= 0)
    
    # define objectives
    model.objective0 = Objective(expr = model.x1 + model.x3)
    model.objective1 = Objective(expr = model.x2 - exp(model.x3))
    
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

# set up options
options = structure()
options.solve_direct = False
options.show_plots = True
options.gap_tolerance = 5e-2
options.nlp_feasibility_only = False
options.constraint_tolerance = 1e-2
options.adaptive_refinement = True
# options.bound_tightening = 3

encl_dict, it = compute_enclosure(build_model, parameter, options)

N, A, lubs, box_list = MOMIBB_direct(
    build_model,
    parameter,
    options,
    encl_dict['zu'],
    encl_dict['zl']
    )