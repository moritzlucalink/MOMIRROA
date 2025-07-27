#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 18 21:37:29 2025

@author: moritz
"""

import numpy as np
from pyomo.environ import *

def compute_ideal_point(call_model, m, box_info, timeout, solver):
    """
    routine for computing the ideal point of a given preimage-space box

    Parameters
    ----------
    call_model : function
        retruning a pyomo model of the problem to be solved.
    m : integer
        representing the number of objective functions.
    box_info : dict
        having the variables as keys and a list of corresponding lower and
        upper variable bounds as values.
    timeout : integer
        representing the timelimit for the method.
    solver : string
        determining which solver should be used for solving the problems.

    Returns
    -------
    ideal_point : array
        representing the ideal point of the given preimage-space box.

    """
    ideal_point = np.zeros(m)   # intialize ideal point
    
    for i in range(m):
        model = call_model(i)   # intialize model
        
        for v in model.component_objects(Var):
            v.lb = box_info[v.name][0]
            v.ub = box_info[v.name][1]
            
        opt = SolverFactory(solver)
        
        if solver == 'gurobi':
            opt.options['TimeLimit'] = timeout
        elif solver == 'scip':
            opt.options['limits/time'] = timeout
        
        results = opt.solve(model, tee=False, options={'threads': 1})
        
        if results.solver.termination_condition == TerminationCondition.optimal:
            for o in model.component_objects(Objective):
                if o.active:
                    ideal_point[i] = value(o)
        elif results.solver.termination_condition == TerminationCondition.infeasible:
            print('detected infeasible box')
            return None
    
    return ideal_point
