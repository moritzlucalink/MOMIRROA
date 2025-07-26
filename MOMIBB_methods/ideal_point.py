#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 18 21:37:29 2025

@author: moritz
"""

import numpy as np
from pyomo.environ import *

def compute_ideal_point(call_model, m, box_info, timeout, solver):
    
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
