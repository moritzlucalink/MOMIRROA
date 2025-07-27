#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 23:42:50 2025

@author: moritz
"""

from pyomo.environ import *

import copy as cp
import numpy as np


def compute_least_square_weight(cons, vars, info):
    """
    routine for computing the least square weight of a linear function with
    respect to the corner points of a variable domain box and a constraint
    function

    Parameters
    ----------
    cons : pyomo constraint object
        representing the constraint function for which the least square weight
        should be computed.
    vars : list
        containing pyomo variable objects appearing in the constraint function
        of interest.
    info : dict
        containing all information of the current box of interest.

    Returns
    -------
    weight : dict
        containing respective weights for all appearing variables as well as
        the constant term.

    """
    
    n = len(vars)
    
    # extract lower/upper bounds for each var from info
    lbs = np.array([min(info[v.name]) for v in vars], dtype=float)
    ubs = np.array([max(info[v.name]) for v in vars], dtype=float)
    
    # generate all 2^n corner points
    P = 1 << n
    idx = np.arange(P)[:, None]
    bits = (idx >> np.arange(n)) & 1
    corners = np.where(bits, ubs, lbs)
    
    # build matrix
    A = np.hstack((corners, np.ones((P, 1))))
    
    # evaluate b by fixing each corner
    b = np.empty(P, dtype=float)
    for i in range(P):
        # assign corner i
        for j, v in enumerate(vars):
            v.fix(corners[i, j], skip_validation=True)
        b[i] = value(cons)
    
    # unfix all variables
    for v in vars:
        v.unfix()
        
    # solve least squares, fallback to all-ones on failure
    try:
        w_vec = np.linalg.lstsq(A, b, rcond=None)[0]
    except:
        w_vec = np.ones(n + 1)
        
    # round and package into dict
    w_vec = np.round(w_vec, 6)
    weight = {v.name: w_vec[j] for j, v in enumerate(vars)}
    weight['constant'] = w_vec[-1]
    
    return weight
    
    

