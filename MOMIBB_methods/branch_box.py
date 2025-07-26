#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 15:27:02 2025

@author: moritz

Branching module for subdividing a preimage-space box by its longest edge.

Provides the 'branch_box' function to split a box into two smaller boxes.
"""

# from copy import deepcopy
# from numpy import floor, ceil
# from typing import Dict, Tuple, Union

# Numeric = Union[int, float]
# Bounds = Tuple[Numeric, Numeric, str]
# BoxInfo = Dict[str, Bounds]

# __all__ = ["branch_box"]

# def branch_box(box: BoxInfo) -> Tuple[BoxInfo, BoxInfo]:
#     """
#     Split the input box along its longest edge.

#     Parameters
#     ----------
#     box : BoxInfo
#         a dictionary mapping dimension names to a tuple
#         (lower_bound, upper_bound, var_type), where 
#         var_type is 'Reals' for continuous or 'Integers' or 'Binary' for discrete.

#     Returns
#     -------
#     Tuple[BoxInfo, BoxInfo]
#         a tuple (box1, box2), where box1 and box2 are independent copies of the input box
#         with the longest edge divided at its midpoint.

#     """
#     # create deep copies to avoid mutating the original
#     box1 = deepcopy(box)
#     box2 = deepcopy(box)
    
#     # identify the longest edge by its length
#     longest_edge = max(
#         box,
#         key=lambda k: box[k][1] - box[k][0]
#     )
#     lower, upper, var_type = box[longest_edge]
#     length = upper - lower

#     # compute midpoint
#     midpoint = (lower + upper) / 2
    
#     # split based on variable typ
#     if var_type != 'Reals': # discrete variable
#         # handle odd-length intervals with floor/ceil to preserve integer bounds
#         if length % 2 == 1:
#             box1[longest_edge] = (lower, int(floor(midpoint)), var_type)
#             box2[longest_edge] = (int(ceil(midpoint)), upper, var_type)
#         else:
#             box1[longest_edge] = (lower, int(midpoint), var_type)
#             box2[longest_edge] = (int(midpoint), upper, var_type)
#     else: # continuous variable
#         box1[longest_edge] = (lower, midpoint, var_type)
#         box2[longest_edge] = (midpoint, upper, var_type)
        
#     return box1, box2


# # ---------------------
# # Unit Tests (pytest)
# # ---------------------

# def test_branch_box_continuous():
#     box = {'x': (0.0, 10.0, 'Reals')}
#     b1, b2 = branch_box(box)
#     assert b1['x'] == (0.0, 5.0, 'Reals')
#     assert b2['x'] == (5.0, 10.0, 'Reals')


# def test_branch_box_discrete_even():
#     box = {'i': (0, 10, 'Ints')}
#     b1, b2 = branch_box(box)
#     # Length 10 even -> midpoint = 5
#     assert b1['i'] == (0, 5, 'Ints')
#     assert b2['i'] == (5, 10, 'Ints')


# def test_branch_box_discrete_odd():
#     box = {'j': (-9, 0, 'Ints')}
#     b1, b2 = branch_box(box)
#     # Length 9 odd -> lower half 0..4, upper half 5..9
#     assert b1['j'] == (-9, -5, 'Ints')
#     assert b2['j'] == (-4, 0, 'Ints')


# def test_branch_box_multidim():
#     box = {
#         'x': (0, 5, 'Reals'),
#         'y': (0, 10, 'Reals')
#     }
#     b1, b2 = branch_box(box)
#     # Longest edge is y
#     assert b1['y'] == (0, 5.0, 'Reals')
#     assert b2['y'] == (5.0, 10, 'Reals')
#     # Other dimension unchanged
#     assert b1['x'] == (0, 5, 'Reals')
#     assert b2['x'] == (0, 5, 'Reals')

# def test_branch_box_multidim_samelength():
#     box = {
#         'x': (0, 5, 'Reals'),
#         'y': (0, 5, 'Reals')
#     }
#     b1, b2 = branch_box(box)
#     # Longest edge is x (first to appear)
#     assert b1['x'] == (0, 2.5, 'Reals')
#     assert b2['x'] == (2.5, 5, 'Reals')
#     # Other dimension unchanged
#     assert b1['y'] == (0, 5, 'Reals')
#     assert b2['y'] == (0, 5, 'Reals')


# if __name__ == "__main__":
#     import pytest
#     pytest.main([__file__])


import copy as cp
import numpy as np

def branch_box(box_info):
    
    new_box1 = cp.deepcopy(box_info)
    new_box2 = cp.deepcopy(box_info)
    
    # catch longest edge
    longest_edge_len = 0
    for key in box_info.keys():
        length = box_info[key][1] - box_info[key][0]
        if length > longest_edge_len:
            longest_edge_len = length
            longest_edge = key
            
    # compute middle point
    mid = (box_info[longest_edge][0] + box_info[longest_edge][1]) / 2
    
    # box updates
    # start with integers
    if box_info[longest_edge][2] != 'Reals':
        if longest_edge_len%2 == 1:
            new_box1[longest_edge][1] = int(np.floor(mid))
            new_box2[longest_edge][0] = int(np.ceil(mid))
            
        else:
            new_box1[longest_edge][1] = int(mid)
            new_box2[longest_edge][0] = int(mid)
    
    # for continuous variables
    else:
        new_box1[longest_edge][1] = mid
        new_box2[longest_edge][0] = mid
    
    return new_box1, new_box2
        