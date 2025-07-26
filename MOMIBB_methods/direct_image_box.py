#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 13:40:54 2024

@author: moritz

Direct image box module: computes a bounding box of the multi-objective image set
by solving each objective in turn, without relying on relaxations.
"""

# from typing import Callable, Tuple, Optional
# import numpy as np

# from pyomo.environ import SolverFactory, Objective, maximize, value, ConcreteModel
# from rounding_routines import rounding_lower, rounding_upper

# __all__ = ['compute_direct_image_box']

# SolverName = str
# BuildModel = Callable[[int], ConcreteModel]


# def compute_direct_image_box(
#         build_model: BuildModel,
#         num_objectives: int,
#         tol: Optional[float] = None,
#         solver_name: SolverName = 'scip'
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Compute an axis-aligned box that strictly contains the image of a multi-objective
#     model by sequentially minimizing/maximizing each objective.

#     Parameters
#     ----------
#     build_model : BuildModel
#         callable mapping an objective index to a Pyomo ConcreteModel.
#     num_objectives : int
#         number of objectives (dimension of output space).
#     tol : Optional[float], optional
#         solver gap tolerance (for SCIP or Gurobi). The default is None, no customized gap limit is set.
#     solver_name : SolverName, optional
#         which solver to use ('scip' or 'gurobi'). The default is 'scip'.

#     Returns
#     -------
#     zl : ndarray
#         lower bounds array of length num_objectives.
#     zu : ndarray
#         upper bounds array of length num_objectives.

#     """
#     # initialize containers for ideal and nadir points
#     zl = np.zeros(num_objectives)
#     zu = np.zeros(num_objectives)
    
#     def _solve(obj_index: int, maximize_obj: bool) -> float:
        
#         # build fresh model for each solve
#         model = build_model(obj_index)
        
#         # set objective sense
#         for obj in model.component_objects(Objective, active=True):
#             obj.sense = maximize if maximize_obj else obj.sense
        
#         # configure solver
#         solver = SolverFactory(solver_name)
#         if tol is not None:
#             if solver_name == 'scip':
#                 solver.options['limits/gap'] = tol
#             elif solver_name == 'gurobi':
#                 solver.options['MIPGap'] = tol
                
#         results = solver.solve(model, tee=False)
        
#         # extract and adjust bound/value
#         if solver_name == 'scip':
#             bound = results.solver.dual_bound
#             return(rounding_upper(bound, 10) if maximize_obj
#                    else rounding_lower(bound, 10))
#         elif solver_name == 'gurobi':
#             val = next(value(o) for o in model.component_objects(Objective) if o.active)
#             return (val + tol if maximize_obj else val - tol)
        
#     # compute ideal point (minimization)
#     for idx in range(num_objectives):
#         zl[idx] = _solve(idx, maximize_obj=False)
    
#     # compute nadir point (maximization)
#     for idx in range(num_objectives):
#         zu[idx] = _solve(idx, maximize_obj=True)
        
#     return zl, zu


from pyomo.environ import *
from pyomo.core.expr import identify_variables

import numpy as np

from rounding_routines import *

def direct_image_box(call_model, m, tol=False, solver='scip'):
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
            elif solver == 'gurobi':
                opt.options['MIPGap'] = tol
        
        results = opt.solve(model, tee=False)
        
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
            elif solver == 'gurobi':
                opt.options['MIPGap'] = tol
        
        results = opt.solve(model, tee=False)
        
        # catch optimal value
        if solver == 'scip':
            zu[i] = rounding_upper(results.solver.dual_bound,10)
        elif solver == 'gurobi':
            for o in model.component_objects(Objective):
                if o.active == True:
                    zu[i] = value(o) + tol
    return zl, zu