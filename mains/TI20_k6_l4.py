#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:20:11 2024

@author: moritz
"""

from pyomo.environ import *
from pyomo.core.expr.current import identify_variables

import sys

sys.path.append('../MOMIRROA_methods')
sys.path.append('../MOMIBB_methods')

from MOMIRROA import *
from MOMIBB_direct import *

"""
problem instance (P3) with k=6 and l=4 from
Eichfelder, G., Stein, O., and Warnow, L. A Solver For Multiobjective
Mixed-Integer Convex and Nonconvex Optimization. 2023

"""

plt.close('all')

class structure():
    pass


# define the model
def build_model(m):

    model = ConcreteModel()

    # define variables
    model.x1 = Var(within=Reals, bounds=(0,1))
    model.x2 = Var(within=Reals, bounds=(0,1))
    model.x3 = Var(within=Reals, bounds=(0,1))
    model.x4 = Var(within=Reals, bounds=(0,1))
    model.x5 = Var(within=Reals, bounds=(0,1))
    model.x6 = Var(within=Reals, bounds=(0,1))
    model.x7 = Var(within=Integers, bounds=(-3,3))
    model.x8 = Var(within=Integers, bounds=(-3,3))
    model.x9 = Var(within=Integers, bounds=(-3,3))
    model.x10 = Var(within=Integers, bounds=(-3,3))

    # define constraints
    model.cons_nonlin0 = Constraint(expr = -model.x1**2 - model.x2**2 - \
                                    model.x3**2 - model.x4**2 - model.x5**2 \
                                        - model.x6**2 + 1 <= 0)
    model.cons_nonlin1 = Constraint(expr = model.x7**2 + model.x8**2 \
                                    + model.x9**2 + model.x10**2 - 9 <= 0)

    # define objectives
    model.objective0 = Objective(expr = model.x1 + model.x2 + model.x3 +\
                                 model.x7 + model.x8)
    model.objective1 = Objective(expr = model.x4 + model.x5 + model.x6 +\
                                 model.x9 + model.x10)

    for o in model.component_objects(Objective):
        if not 'objective'+str(m) in o.name:
            o.deactivate()

    return model

# set up the parameters
parameter = structure()
parameter.m = 3
parameter.tol = 0.1
parameter.maxiter = 5000
parameter.timeout = 3600
parameter.factor_delta = 0.95 * parameter.tol

# set up options
options = structure()

options.solve_direct = False # determine if relaxations should be used

options.gap_tolerance = 1e-4 # determine the gap if no relaxations are used

options.show_plots = True # determine if plots should be shown

options.nlp_feasibility_only = True # determine if NLP should be solved only to feasibility

options.constraint_tolerance = 1e-6 # determine constraint violation tolerance

options.adaptive_refinement = False # determine if adaptive refinement procedure should be applied

options.bound_tightening = 1e20 # determine if (and when) OBBT should be applied

options.soft_utopian_check = True # determine if new utopian should be required to be not included in current utopians

options.method = 'Nothing' # determine which method should be applied


if options.method == 'MOMIRROA':
    print('MOMIRROA is chosen for computing an enclosure')
    encl_dict, it = MOMIRROA(build_model, parameter, options)

elif options.method == 'MOMIBB':
    print('MOMIBB direct is chosen for computing an enclosure')
    encl_dict, it = MOMIBB_direct(build_model, parameter, options)