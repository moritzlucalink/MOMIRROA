#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 23:08:13 2025

@author: moritz
"""

import copy as cp
import numpy as np

def update_lub_rel_info(encl_dict, info, u, old_lubs):
    """
    routine for assigning preimage space relaxation information to the incoming
    local upper bounds

    Parameters
    ----------
    encl_dict : dict
        containing all information collected by the algorithm.
    info : dict
        containing all informatio for setting up the current piecewise linear
        relaxation of the problem of interest.
    u : array
        representing the update point.
    old_lubs : list
        representing the list of old local upper bounds.

    Returns
    -------
    new_info : dict
        having the currently active lubs as keys and the corresponding
        relaxation information dicts as values.

    """
    
    # backup the old info map
    old_info = cp.deepcopy(encl_dict.get('lub_relaxation_information', {}))
    
    # prepare arrays of new lubs
    lubs_list = [np.asarray(l) for l in encl_dict['lubs']]
    m = len(lubs_list)
    
    if m == 0:
        encl_dict['lub_relaxation_information'] = {}
        return {}
    
    n = lubs_list[0].size
    lub_arr = np.stack(lubs_list, axis=0)
    
    # which lubs have already stored info?
    key_exists = np.array([str(l) in old_info for l in lubs_list])
    
    # which lubs lie entirely below or on u?
    u = np.asarray(u)
    below_u = (lub_arr <= u).all(axis=1)
    
    # prepare old_lbs array for 'covered by old' check
    old_arr = (np.stack([np.asarray(o) for o in old_lubs], axis=0) \
        if old_lubs
        else np.empty((0,n)))
        
    # build new info dict
    new_info = {}
    
    # copy over any existing entries wholesale
    for i, lub in enumerate(lubs_list):
        key = str(lub)
        if key_exists[i]:
            new_info[key] = cp.deepcopy(old_info[key])
            
        # handle those that need fresh assignment
        else:
            # directly below u? take 'info' and bump counter
            if below_u[i]:
                new_info[key] = cp.deepcopy(info)
                new_info[key]['BT counter'] += 1
                
            else:
                # find any old_lub that dominates this lub
                if old_arr.size > 0:
                    dominates = (lub <= old_arr).all(axis=1)
                    idx = np.argmax(dominates) if dominates.any() else None
                    if idx is not None and dominates[idx]:
                        old_key = str(old_lubs[idx])
                        new_info[key] = cp.deepcopy(old_info[old_key])
                        new_info[key]['BT counter'] += 1
                        continue
                
    return new_info
                    
            

