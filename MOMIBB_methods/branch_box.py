#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 15:27:02 2025

@author: moritz

Branching module for subdividing a preimage-space box by its longest edge.

Provides the 'branch_box' function to split a box into two smaller boxes.
"""

import copy as cp
import numpy as np

def branch_box(box_info):
    """
    routine for performing a longest edge bisection of the longest edge of a
    preimage-space box

    Parameters
    ----------
    box_info : dict
        having the variable names as keys and a list of the corresponding lower
        and upper bounds as values.

    Returns
    -------
    new_box1 : dict
        representing the first box after longest edge bisection.
    new_box2 : dict
        representing the second box after longest edge bisection.

    """
    
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
        