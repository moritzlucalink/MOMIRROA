#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 20:32:34 2025

@author: moritz
"""


from pyomo.environ import *
from pyomo.core.expr.current import identify_variables, identify_components

import sys

sys.path.append('../MOMIRROA_methods')
sys.path.append('../MOMIBB_methods')

from MOMIRROA import *
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
    model.x1 = Var(within=Reals, bounds=(0,100))


    # define objectives
    model.objective0 = Objective(expr = model.x1)
    model.objective1 = Objective(expr = model.x1**2)

    for o in model.component_objects(Objective):
        if not 'objective'+str(m) in o.name:
            o.deactivate()

    return model

# set up the parameters
parameter = structure()
parameter.m = 2
parameter.tol = 0.1
parameter.maxiter = 2000
parameter.timeout = 3600
parameter.factor_delta = 0.95 * parameter.tol

# set up options
options = structure()

options.solve_direct = False # determine if relaxations should be used

options.gap_tolerance = 1e-4 # determine the gap if no relaxations are used

options.show_plots = True # determine if plots should be shown

options.nlp_feasibility_only = False # determine if NLP should be solved only to feasibility

options.constraint_tolerance = 1e-6 # determine constraint violation tolerance

options.adaptive_refinement = False # determine if adaptive refinement procedure should be applied

options.bound_tightening = 1e20 # determine if (and when) OBBT should be applied

options.soft_utopian_check = True # determine if new utopian should be required to be not included in current utopians

options.method = 'MOMIRROA' # determine which method should be applied


if options.method == 'MOMIRROA':
    print('MOMIRROA is chosen for computing an enclosure')
    encl_dict, it = MOMIRROA(build_model, parameter, options)

elif options.method == 'MOMIBB':
    print('MOMIBB direct is chosen for computing an enclosure')
    encl_dict, it = MOMIBB_direct(build_model, parameter, options)
