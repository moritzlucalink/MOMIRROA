#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 21:41:27 2025

@author: moritz
"""

import numpy as np


def update_nondom(N, y):
    
    
    y = np.asarray(y)
    
    # ensure empty case
    if not N:
        return [y.copy()], 1
    
    # stack N into matrix
    N_arr = np.stack(N, axis=0)
    if N_arr.shape[1] != y.size:
        raise ValueError('Shapes of points in N and y do not align')
        
    # if y in N, keep N and set check=0
    eq_mask = np.all(N_arr == y, axis=1)
    if eq_mask.any():
        return N, 0
    
    # which x dominate y?
    dom_mask = np.all(N_arr <= y, axis=1)
    
    # which x are not weakly dominated by y?
    other_mask = np.any(N_arr < y, axis=1) & (~dom_mask)
    
    # keep the survivors, i.e., where dom_mask or other_mask is True
    survivors = N_arr[dom_mask | other_mask]
    new_N = [row.copy() for row in survivors]
             
    # determine check flag
    if dom_mask.any():
        check = 0
    else:
        check = 1
        new_N.append(y.copy())
        
    return new_N, check
