#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 23:32:09 2025

@author: moritz
"""

import copy as cp
import numpy as np

from initialize_local_bound_sets import *
from update_llbs import *


def rebuild_utopian_llbs(encl_dict, y):
    
    # deepcopy old state
    old = cp.deepcopy(encl_dict)
    
    # copy over all static entries
    static_keys = [
        'zl', 'zu', 'dir_vec', 'factor_delta', 'N',
        'analysis', 'start_time', 'timeout',
        'lubs', 'Udefpois', 'lub_relaxation_information'
    ]
    new = {k: cp.deepcopy(old[k]) for k in static_keys}
    
    # reinitialize llbs/Ldefpois (drops in-place any old ones)
    llbs, Ldefpois, _, _ = init_locbounds_defpois(new)
    new['llbs'], new['Ldefpois'] = llbs, Ldefpois
    
    # rebuild U by masking out any x with y <= x
    old_U = old.get('U', [])
    if old_U:
        U_arr = np.stack(old_U, axis=0)
        mask_keep = ~np.all(y[None, :] <= U_arr, axis=1)
        U_arr = U_arr[mask_keep]
    else:
        U_arr = np.empty((0, y.size))
        
    # if y isn't already present, append it
    if not (U_arr.shape[0] and np.any(np.all(U_arr == y, axis=1))):
        U_arr = np.vstack([U_arr, y])
    
    # convert back to list of 1-d arrays
    new['U'] = [row.copy() for row in U_arr]
    
    # sequentially update llbs & LDefpois for each utopian point
    for x in new['U']:
        new['llbs'], new['Ldefpois'] = update_llbs(
            new['llbs'], new['Ldefpois'], x
        )
    
    return new