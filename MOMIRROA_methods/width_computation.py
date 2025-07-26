#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 23:00:20 2025

@author: moritz
"""

import copy as cp
import numpy as np
import sys


def compute_width(llbs, lubs, dir_vec):
    """
    vectorized computation of the relative width of the enclosure determined by
    lubs and llbs w.r.t. the vector dir_vec

    Parameters
    ----------
    llbs : list of 1-d arrays, shape (n,)
    lubs : list of 1-d arrays, shape (n,)
    dir_vec : 1-d array, shape(n,)

    Returns
    -------
    width : float
    worst_llb : 1-d array, shape(n,)
    worst_lub : 1-d array, shape(n,)

    """
    
    # stack lists into matrices
    llb_arr = np.stack(llbs, axis=0)
    lub_arr = np.stack(lubs, axis=0)
    
    # ensure dir_vec shape
    dir_vec = np.asarray(dir_vec)
    
    # compute pairwise differences
    diff = lub_arr[None, :, :] - llb_arr[:, None, :]
    
    # mask boxes where llb < lub in *every* dimension
    valid = (diff > 0).all(axis=2)
    
    # project edge-lengths onto dir_vec and take the minimum over dims
    projected = diff / dir_vec[None, None, :]
    s_vals = projected.min(axis=2)
    
    # invalidate the pairs that weren't boxes
    s_vals[~valid] = -np.inf
    
    # find the maximum width and its indices
    # unravale_index gives (i_ll, j_lub)
    max_idx = np.unravel_index(np.argmax(s_vals), s_vals.shape)
    width = s_vals[max_idx]
    
    # grab the corresponding corners
    worst_llb = llb_arr[max_idx[0], :].copy()
    worst_lub = lub_arr[max_idx[1], :].copy()
    
    return width, worst_llb, worst_lub