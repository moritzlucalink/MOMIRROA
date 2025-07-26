#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 00:10:01 2025

@author: moritz

Feasible point module: finds a feasible point belonging to a preimage-sapce box
for multi-objective models via the Pascoletti-Serafini scalarization.
"""

# from typing import Callable, Dict, Tuple, Union, Optional
# import numpy as np
# from pyomo.environ import (
#     SolverFactory,
#     Objective,
#     Var,
#     Constraint,
#     Reals,
#     value,
#     ConcreteModel
# )

# __all__ = ['compute_feasible_point']

# BuildModel = Callable[[int], ConcreteModel]
# Numeric = Union[int, float]
# Bounds = Tuple[Numeric, Numeric, str]
# BoxInfo = Dict[str, Bounds]


# def compute_feasible_point(
#         build_model: BuildModel,
#         num_objectives: int,
#         box_info: BoxInfo,
#         ideal_point: np.ndarray,
#         timeout: float,
#         solver_name: str
# )   ->  np.ndarray:
#     """
#     Find a feasible point within a given box using Pascoletti-Serafini scalarization.

#     Parameters
#     ----------
#     build_model : BuildModel
#         Callable mapping an objective index to a Pyomo model.
#     num_objectives : int
#         number of objectives (dimension of imgae space).
#     box_info : BoxInfo
#         a dictionary mapping dimension names to a tuple
#         (lower_bound, upper_bound, var_type), where 
#         var_type is 'Reals' for continuous or 'Integers' or 'Binary' for discrete..
#     ideal_point : np.ndarray
#         array representing the ideal point (objectives minimized)..
#     timeout : float
#         solver time limit (seconds).
#     solver_name : str
#         which solver to use ('scip' or 'gurobi').

#     Returns
#     -------
#     image_vector : np.ndarray
#         array of objective values of the feasible point.

#     """
#     # initialize the model
#     model = build_model(0)
    
#     # apply variable bounds
#     for var in model.component_objects(Var):
#         var.lb = box_info[var.name][0]
#         var.ub = box_info[var.name][1]
        
#     # define scalarization parameter
#     model.t = Var(within=Reals, bounds=(-1e20, 1e20))
    
#     # add pascoletti-serafini constraints and deactivate original objectives
#     for obj in model.component_objects(Objective):
#         for idx in range(num_objectives):
#             if 'objective'+str(idx) in obj.name:
#                 model.add_component(
#                     obj.name+'ps_cons',
#                     Constraint(expr = 0 <= ideal_point[idx] + model.t - obj.expr)
#                 )
#                 obj.deactivate()
        
#         # name = obj.name
#         # if any(f'objective{i}' in name for i in range(num_objectives)):
#         #     idx = int(name.replace('objective', ''))
#         #     model.add_component(
#         #         name+'ps_cons',
#         #         Constraint(expr = 0 <= ideal_point[idx] + model.t - obj.expr)
#         #     )
#         #     obj.deactivate()
            
#     # new objective: minimize t
#     model.scalar_obj = Objective(expr = model.t)
    
#     # configure solver
#     solver = SolverFactory(solver_name)
#     if solver_name == 'gurobi':
#         solver.options['TimeLimit'] = timeout
#     elif solver_name == 'scip':
#         solver.options['limits/time'] = timeout
        
#     # solve
#     results = solver.solve(model, tee=False)
    
#     # extract image vector from original objectives
#     image_vector = np.zeros(num_objectives)
#     for obj in model.component_objects(Objective, active=False):
#         for idx in range(num_objectives):
#             if 'objective'+str(idx) in obj.name:
#                 image_vector[idx] = value(obj)
        
        
#         # name = obj.name
#         # if any(f'objective{i}' in name for i in range(num_objectives)):
#         #     idx = int(name.replace('objective', ''))
#         #     image_vector[idx] = value(obj)
            
#     return image_vector




from pyomo.environ import *

import numpy as np

def compute_feasible_point(call_model, m, box_info, ideal_point, timeout, solver):
    
    
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