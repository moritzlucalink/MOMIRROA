#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 18 22:54:37 2025

@author: moritz

Module for testing whether a preimage-space box can be discarded
based on local upper bounds (LUBs).
"""

# from typing import Callable, Dict, List, Tuple, Union
# import numpy as np
# from ps_discard import ps_discard

# __all__ = ['discarding_test']

# BuildModel = Callable[[int], object]
# Numeric = Union[int, float]
# Bounds = Tuple[Numeric, Numeric, str]
# BoxInfo = Dict[str, Bounds]

# def discarding_test(
#         build_model: BuildModel,
#         box_info: BoxInfo,
#         ideal_point: np.ndarray,
#         lubs: List[np.ndarray],
#         timeout: float,
#         solver: str,
#         count: int
# )   ->  Tuple[bool, int]:
#     """
#     Determine if a box can be discarded by comparing its ideal point
#     to a list of local upper bounds and optionally running a more elaborate test.

#     Parameters
#     ----------
#     build_model : BuildModel
#         callable mapping an objective index to a Pyomo ConcreteModel.
#     box_info : BoxInfo
#         a dictionary mapping dimension names to a tuple
#         (lower_bound, upper_bound, var_type), where 
#         var_type is 'Reals' for continuous or 'Integers' or 'Binary' for discrete.
#     ideal_point : np.ndarray
#         array representing the ideal point (objectives minimized).
#     lubs : List[np.ndarray]
#         list of arrays representing known local upper bounds.
#     timeout : float
#         timelimit for elaborate discarding test.
#     solver : str
#         solver name to use for elaborate discarding test.
#     count : int
#         count of solved MINLPs.

#     Returns
#     -------
#     keep : Boolean
#         True if the box should be kept; False if discarded.
#     count : int
#         total number of MINLPs solved.
#     """

#     # stack lubs into a 2D array
#     lub_array = np.vstack(lubs)
#     # mask lubs that are dominated by the ideal point
#     mask = np.all(ideal_point <= lub_array, axis=1)
#     relevant_lubs = lub_array[mask]
    
#     if relevant_lubs.size == 0:
#         print('discarded by trivial test')
#         return False, count
    
#     # try elaborate test against each relevant lub
#     for lub in relevant_lubs:
#         count += 1
#         keep = ps_discard(build_model, box_info, lub, timeout, solver)
#         if keep:
#             return True, count

#     # if none kept the box
#     print('discarded by elaborate test')
#     return False, count




from ps_discard import *

import numpy as np

def discarding_test(call_model, box_info, ideal_point, lubs, timeout, solver, count):
    
    lub_arr = np.stack(lubs, axis=0)
    affected_lubs_mask = np.all(ideal_point[None,:] <= lub_arr, axis=1)
    affected_lubs = lub_arr[affected_lubs_mask]
    
    if len(affected_lubs) == 0:
        print('discarded by trivial test')
        return False, count
    
    keep = False
    
    for lub in affected_lubs:
        keep = ps_discard(call_model, box_info, lub, timeout, solver)
        
        count += 1

        if keep:
            return keep, count
        
    if not keep:
        print('discarded by elaborate test')
    
    return keep, count
    