#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 21:30:01 2025

@author: moritz
"""

import numpy as np

def update_utopian(U, y, options):
    """
    update the stable set of utopian points w.r.t. a new utopian point
    
    for references see Algorithm 4 in 
    
    Link, M. and Volkwein, S. 
    Adaptive piecewise linear relaxations for enclosure computations for
    nonconvex multi-objective mixed-integer quadratically constrained programs
    J. Global Optim. 87. 2023.

    Parameters
    ----------
    U : list
        consisting of a stable set of utopian points.
    y : array
        the new utpoian point.
    options : structure
        consisting of possible options for procedure.

    Raises
    ------
    ValueError
        if shapes of new utopian point and old ones do not align.

    Returns
    -------
    new_U : list
        updated list of utopian points.
    check : boolean
        indicating if y belongs to updated list of utopian points, i.e., if y
        improved U.

    """
    
    y = np.asarray(y)
    n = y.size
    
    # get soft-check flag
    soft = getattr(options, 'soft_utopian_check', False)
    
    # empty-set case
    if not U:
        return [y.copy()], 1
    
    # stack current utopians into an (m, n) array
    U_arr = np.stack(U, axis=0)
    if U_arr.shape[1] != n:
        raise ValueError('shapes do not align between U points and y')
        
    # early exit if y exactly equals any existing point
    eq = np.all(U_arr == y, axis=1)
    if eq.any():
        return U, int(soft)
    
    # compute bulk masks
    # mask_dom:  y <= x in ALL dims 
    # mask_strict: y < x in ANY dim
    mask_dom = np.all(y <= U_arr, axis=1)
    mask_strict = np.any(y < U_arr, axis=1)
    keep = mask_dom | mask_strict
    
    # collect survivors
    new_U = [row.copy() for row in U_arr[keep]]
    
    # determine check flag and maybe add y
    if mask_dom.any():
        check = 0
    else:
        check = 1
        new_U.append(y.copy())
        
    return new_U, check