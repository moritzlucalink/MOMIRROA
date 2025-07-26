#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 18 23:44:13 2025

@author: moritz
"""

import numpy as np

def update_lowerbounds(A, new_lb):
    
    if len(A) == 0:
        return [new_lb]
    
    A_arr = np.stack(A, axis=0)
    add_check = np.all(A <= new_lb[None,:], axis=1)
    
    if not any(add_check):
        
        affected_llbs = np.all(new_lb[None,:] <= A, axis=1)
        new_A = A_arr[~affected_llbs]
        A = [row.copy() for row in new_A]
        A.append(new_lb)
    
    return A