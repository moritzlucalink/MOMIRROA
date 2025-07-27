#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 22:28:30 2025

@author: moritz
"""

import copy as cp
import numpy as np


def update_llbs(llbs, defpois, z):
    """
    routine for updating the set of local lower bounds with corresponding
    defining points w.r.t. a new point z
    
    Adapted version for local lower bounds of Algorithm 5 in 
    
    Klamroth, K. and Lacour, R. and Vanderpooten, D.
    On the representation of the search region in multi-objective optimization.
    Eur. J. Oper. Res. 245(3). 2015.

    Parameters
    ----------
    llbs : list
        representing the current assignment of local lower bounds.
    defpois : dict
        with llbs as keys in first layer and indices as keys in second layer
        and list of defining points as values in third layer.
    z : array
        representing the update point.

    Raises
    ------
    ValueError
        if shapes of new update point and old local lower bounds do not align.

    Returns
    -------
    new_llbs : list
        representing the updated assignment of local lower bounds.
    new_defpois : dict
        representing the updated assignment of defining points.

    """
    
    z = np.asarray(z)
    n = z.size
    
    # cop old defining points dict
    defpois_old = cp.deepcopy(defpois)
    
    # stack llbs into an (m, n) array
    llb_arr = np.stack(llbs, axis=0)
    if llb_arr.shape[1] != n:
        raise ValueError('Shapes of llbs and z do not align')
        
    # find affected llbs A where llb < z 
    A_mask = (llb_arr < z).all(axis=1)
    
    # face-touch updates: if llb[i,j] == z[j] but
    # llb < z for all other dims, append z to defpois for that j
    eq_mask = (llb_arr == z[None, :])
    less_mask = (llb_arr < z[None, :])
    for j in range(n):
        others_lt = less_mask[:, [k for k in range(n) if k!=j]].all(axis=1)
        hits = np.nonzero(eq_mask[:, j] & others_lt)[0]
        for i in hits:
            key = str(llbs[i])
            defpois[key][j].append(z.copy())
            
    # build new candidate set P by splitting each affected llb
    P = []
    for i in np.nonzero(A_mask)[0]:
        llb = llbs[i]
        old_defs = defpois_old[str(llb)]
        for j in range(n):
            # for each k!=j, compute max over p[j] for p in old_defs[k]
            zmax_list = [
                max(p[j] for p in old_defs[k])
                for k in range(n) if k!=j
            ]
            zmin = min(zmax_list)
            if z[j] < zmin:
                # create new lower bound by replacing component j
                lj = llb.copy()
                lj[j] = z[j]
                P.append(lj)
                
                # initialize its defining points entries
                new_defs = {j: [z.copy()]}
                for k in range(n):
                    if k == j:
                        continue
                    
                    # keep only those old points whose j-th component > z[j]
                    new_defs[k] = [
                        p.copy()
                        for p in old_defs[k]
                        if p[j] > z[j]
                    ]
                defpois[str(lj)] = cp.deepcopy(new_defs)
                
    # survivors are original llbs not in A
    survivors = llb_arr[~A_mask]
    new_llbs = [u.copy() for u in P] + [row.copy() for row in survivors]
    
    # build new defpois dict for exactly the new_llbs
    new_defpois = {
        str(lb): cp.deepcopy(defpois[str(lb)])
        for lb in new_llbs
    }
    
    return new_llbs, new_defpois
    
