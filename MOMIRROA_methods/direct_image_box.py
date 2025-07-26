#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 13:40:54 2024

@author: moritz
"""

from pyomo.environ import *
from pyomo.core.expr import identify_variables

import numpy as np

from rounding_routines import *

def direct_image_box(call_model, m, tol=False, solver='scip', timelimit=3600):
    """
    routine for computing a box containing the image set without using
    relaxations

    Parameters
    ----------
    call_model : function
        returning a pyomo model of the problem to be solved.
    m : int
        representing the number of objective functions/dimension of the image
        space.
    tol : boolean/float, optional
        indicating if gap tolerance should be adapted. The default is False.
    solver : str, optional
        determining the solver to be used for solving the problem. The default 
        is 'scip'.

    Returns
    -------
    zl : array
        representing the lower left corner of the image box.
    zu : array
        representing the upper right corner of the image box.

    """
    
    # initialize the vectors zl and zu
    zl = np.zeros(m)
    zu = np.zeros(m)
    
    # offset factor to guarantee strict inclusion of image set
    factor = 0.02
    
    # compute ideal point
    for i in np.arange(0,m):
        model = call_model(i)
        
        opt = SolverFactory(solver)
        if tol:
            if solver == 'scip':
                opt.options['limits/gap'] = tol
                opt.options['limits/time'] = timelimit
            elif solver == 'gurobi':
                opt.options['MIPGap'] = tol
                opt.options['TimeLimit'] = timelimit
        
        results = opt.solve(model, tee=False, options={'threads': 1})
        
        # catch optimal value
        if solver == 'scip':
            zl[i] = rounding_lower(results.solver.dual_bound,10)
        elif solver == 'gurobi':
            for o in model.component_objects(Objective):
                if o.active == True:
                    zl[i] = value(o) - tol
        
    # compute reversed ideal point
    for i in np.arange(0,m):
        model = call_model(i)
        
        # maximization
        for o in model.component_objects(Objective):
            if o.active == True:
                o.sense = maximize
                
        opt = SolverFactory(solver)
        if tol:
            if solver == 'scip':
                opt.options['limits/gap'] = tol
                opt.options['limits/time'] = timelimit
            elif solver == 'gurobi':
                opt.options['MIPGap'] = tol
                opt.options['TimeLimit'] = timelimit
        
        results = opt.solve(model, tee=False, options={'threads' : 1})
        
        # catch optimal value
        if solver == 'scip':
            zu[i] = rounding_upper(results.solver.dual_bound,10)
        elif solver == 'gurobi':
            for o in model.component_objects(Objective):
                if o.active == True:
                    zu[i] = value(o) + tol
    
    return zl, zu