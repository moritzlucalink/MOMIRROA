#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 22:10:46 2025

@author: moritz
"""

import copy as cp
import numpy as np

def update_lubs(lubs, defpois, z):
    
    z = np.asarray(z)
    n = z.size
    
    # copy old defpois
    defpois_old = cp.deepcopy(defpois)
    
    # stack lubs into an (m, n) array
    lub_arr = np.stack(lubs, axis=0)
    if lub_arr.shape[1] != n:
        raise ValueError('Shapes of lubs and z do not align')
        
    # find affected lubs where z < lub in every dimension
    A_mask = np.all(z[None, :] < lub_arr, axis=1)
    
    # face-touch updates: for any lub[i],dim j where z[j] == lub[i,j]
    # and in all other dims z < lub, we append z to defpois[lub[i]][j]
    eq_mask = (lub_arr == z[None, :])
    less_mask = (z[None, :] < lub_arr)
    for j in range(n):
        # other dims test
        others_lt = less_mask[:, [k for k in range(n) if k!=j]].all(axis=1)
        hits = np.nonzero(eq_mask[:, j] & others_lt)[0]
        for i in hits:
            key = str(lubs[i])
            defpois[key][j].append(z.copy())
            
    # build the new candidates P by splitting each lub in A along every j
    P = []
    for i in np.nonzero(A_mask)[0]:
        lub = lubs[i]
        old_defs = defpois_old[str(lub)]
        for j in range(n):
            # compute for each k!=j, the minimum over all old_defs[k][:,j]
            zmin_list = [min(p[j] for p in old_defs[k])
                         for k in range(n) if k!=j]
            
            zmax = max(zmin_list)
            
            if zmax < z[j]:
                # we get a new upper-bound uj by replacing coordinate j
                uj = lub.copy()
                uj[j] = z[j]
                P.append(uj)
                
                # now initialize its defpois entries
                new_defs = {j: [z.copy()]}
                for k in range(n):
                    if k == j:
                        continue
                    # only keep those old points p in old_defs[k] with p[j] < z[j]
                    new_defs[k] = [p.copy() for p in old_defs[k]
                                   if p[j] < z[j]]
                
                defpois[str(uj)] = cp.deepcopy(new_defs)
                
    # survivores = original lubs that weren't in A
    survivors = lub_arr[~A_mask]
    new_lubs = [u.copy() for u in P] + [row.copy() for row in survivors]
    
    # build new defpois dict 
    new_defpois = {
        str(lub): cp.deepcopy(defpois[str(lub)])
        for lub in new_lubs
    }
    
    return new_lubs, new_defpois
