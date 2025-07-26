#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:01:30 2024

@author: moritz
"""

import numpy as np
import sys

def shortest_edge(x,y,z):
    """
    compute the shortest edge of a box [x,y] determined by two vectors x and y
    in relation to a third vector z    

    Parameters
    ----------
    x : array
        lower vector for determining the box.
    y : array
        upper vector for determining the box.
    z : array
        vector for determining the relation of the edge lengths.

    Returns
    -------
    s : float
        relative shortest edge length of the box [x,y] relative to z.
    index : int
            representing the index of the shortest edge.

    """
    
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # shape checks
    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError(f'Shape mismatch: x{ x.shape }, y{ y.shape }, z{ z.shape }')
        
    # strict positivity of z
    if not np.all(z > 0):
        raise ValueError('All entries of z must be > 0')
        
    # comute ratio one
    ratio = np.abs(y - x) / z
    
    # extract minimu and its index
    idx = int(ratio.argmin())
    s = float(ratio[idx])
    
    return s, idx


# # test the function
# x = np.array([1,2,3,4])
# y = np.array([9,8,7,6])
# z = -1*np.ones(4)

# s, index = shortest_edge(x,y,z)
# print(s, index)    
