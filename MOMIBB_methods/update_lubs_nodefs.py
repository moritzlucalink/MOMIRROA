#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 09:53:32 2025

@author: moritz
"""

import numpy as np

def update_lubs(lubs, y):
    """
    routine for updating set of local upper bounds w.r.t. a new point y
    
    for reference see Algorithm 3 in 
    
    Klamroth, K. and Lacour, R. and Vanderpooten, D.
    On the representation of the search region in multi-objective optimization.
    Eur. J. Oper. Res. 245(3). 2015.

    Parameters
    ----------
    lubs : list
        representing the current assignment of local upper bounds.
    y : array
        representng the updae point.

    Returns
    -------
    list
        representing the updated assignment of local upper bounds.

    """
    lub_arr = np.stack(lubs, axis=0)
    m = len(y)
    
    # find search zones containing y
    affected_mask = np.all(y[None,:] < lub_arr, axis=1)
    affected_lubs = lub_arr[affected_mask]
    
    # face-touch updates: for any lub[i],dim j where z[j] == lub[i,j]
    # and in all other dims z < lub, we append z to defpois[lub[i]][j]
    eq_mask = (lub_arr == y[None, :])
    less_mask = (y[None, :] < lub_arr)
    
    B = {}
    P = {}
    for j in range(m):
        # other dims test
        others_lt = less_mask[:, [k for k in range(m) if k!=j]].all(axis=1)
        hits = np.nonzero(eq_mask[:, j] & others_lt)[0]
        B[str(j)] = []
        P[str(j)] = []
        for i in hits:
            B[str(j)].append(lubs[i])
            
    for u in affected_lubs:
        for j in range(m):
            u_subs = u.copy()
            u_subs[j] = y[j]
            P[str(j)].append(u_subs)
    P_new = []
    for j in range(m):
        Pj = P[str(j)].copy()
        PB = Pj + B[str(j)].copy()
        if len(PB) > 0:
            PB_arr = np.stack(PB, axis=0)
            for u in Pj:
                PB_u_arr_mask = np.all(u[None,:] == PB_arr,axis=1)
                PB_u_arr = PB_arr[~PB_u_arr_mask]
                if len(PB_u_arr) > 0:
                    if not any(np.all(u[None,:] <= PB_u_arr,axis=1)):
                        P_new.append(u)
                else:
                    P_new.append(u)
        
    not_aff_lubs = lub_arr[~affected_mask]
    old_lubs = [row.copy() for row in not_aff_lubs]
    
    return old_lubs + P_new
    
    
