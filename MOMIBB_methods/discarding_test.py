#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 18 22:54:37 2025

@author: moritz

Module for testing whether a preimage-space box can be discarded
based on local upper bounds (LUBs).
"""

from ps_discard import *

import numpy as np

def discarding_test(call_model, box_info, ideal_point, lubs, timeout, solver, count):
    """
    routine for testing whether a preimage-space box can be discarded

    Parameters
    ----------
    call_model : function
        returning a pyomo model of the problem to be solved.
    box_info : dict
        having the variables as keys and a list of corresponding lower and
        upper variable bounds as values.
    ideal_point : array
        representing the ideal point of the current preimage-space box.
    lubs : list
        containing current assignment of local upper bounds.
    timeout : integer
        representing the timelimit for the method.
    solver : string
        determining which solver should be used for solving the problems.
    count : integer
        representing the number of problems that are solved.

    Returns
    -------
    keep: boolean
        indicating if the current preimage-space box should be kept or can be
        discarded.
    count : integer
        representing the number of problems that are solved.

    """
    lub_arr = np.stack(lubs, axis=0)
    affected_lubs_mask = np.all(ideal_point[None,:] <= lub_arr, axis=1)
    affected_lubs = lub_arr[affected_lubs_mask]

    if len(affected_lubs) == 0:
        print('discarded by trivial test')
        return False, count

    keep = False

    for lub in affected_lubs:
        keep = ps_discard(call_model, box_info, lub, timeout, solver)

        count += 1

        if keep:
            return keep, count

    if not keep:
        print('discarded by elaborate test')

    return keep, count
