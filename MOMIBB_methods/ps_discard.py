#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 21:03:49 2025

@author: moritz
"""

from pyomo.environ import *

def ps_discard(call_model, box_info, lub, timeout, solver):
    
    model = call_model(0)
    
    # set variable bounds
    for v in model.component_objects(Var):
        v.lb = box_info[v.name][0]
        v.ub = box_info[v.name][1]
    
    model.t = Var(within=Reals, bounds=(-1e20, 1e20))
    
    # set up pascoletti-serafini constraints
    for o in model.component_objects(Objective):
        for i in range(0,len(lub)):
            if 'objective'+str(i) in o.name:
                model.add_component(
                    o.name+'ps_cons',
                    Constraint(expr = 0 <= lub[i] + model.t - o.expr)
                )
                o.deactivate()
    
    model.new_obj = Objective(expr = model.t)
    
    opt = SolverFactory(solver)
    
    if solver == 'gurobi':
        opt.options['TimeLimit'] = timeout
    elif solver == 'scip':
        opt.options['limits/time'] = timeout
    
    results = opt.solve(model, tee=False, options={'threads': 1})
    
    if results.solver.termination_condition == TerminationCondition.optimal:
        if value(model.new_obj) <= 0:
            return True
        
        else:
            return False
        
    elif results.solver.termination_condition == TerminationCondition.infeasible:
        print('detected infeasible box')
        return None
    
    else:
        print('something weird happened within the discarding test')
        return None


        
    