#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 00:10:01 2025

@author: moritz

Feasible point module: finds a feasible point belonging to a preimage-sapce box
for multi-objective models via the Pascoletti-Serafini scalarization.
"""

from pyomo.environ import *

import numpy as np

def compute_feasible_point(call_model, m, box_info, ideal_point, timeout, solver):
    """
    routine for computing a feasible point belonging to the current 
    preimage-space box

    Parameters
    ----------
    call_model : function
        returning a pyomo model of the problem to be solved.
    m : integer
        determining the number of objective functions.
    box_info : dict
        having the variables as keys and a list o corresponding lower and 
        upper variable bounds as values.
    ideal_point : array
        representing th ideal point of the current preimage-space box.
    timeout : integer
        representin the timelimit for the method.
    solver : string
        determining which solver should be used for solving the problems.

    Returns
    -------
    image_vector : array
        containing the objective values of the found feasible point.

    """    
    
    z = np.random.randint(m)
    model = call_model(0)
    for v in model.component_objects(Var):
        v.lb = box_info[v.name][0]
        v.ub = box_info[v.name][1]
        
    model.t = Var(within=Reals, bounds=(-1e20, 1e20))
    
    # set up pascoletti-serafini constraints
    for o in model.component_objects(Objective):
        for i in range(0,m):
            if 'objective'+str(i) in o.name:
                model.add_component(
                    o.name+'ps_cons',
                    Constraint(expr = 0 <= ideal_point[i] + model.t - o.expr)
                )
                o.deactivate()
    
    model.new_obj = Objective(expr = model.t)
    
    opt = SolverFactory(solver)
    
    if solver == 'gurobi':
        opt.options['TimeLimit'] = timeout
    elif solver == 'scip':
        opt.options['limits/time'] = timeout
    
    results = opt.solve(model, tee=False, options={'threads': 1})
    
    image_vector = np.zeros(m)
    
    for o in model.component_objects(Objective):
        for i in range(m):
            if 'objective'+str(i) in o.name:
                image_vector[i] = value(o)
                
    return image_vector