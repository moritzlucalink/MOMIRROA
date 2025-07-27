#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 15:31:42 2025

@author: moritz
"""

from pyomo.environ import *

def init_box_list(call_model):
    """
    routine for intializing the list of preimage-space boxes that have to be
    treated by the algorithm

    Parameters
    ----------
    call_model : function
        returning a pyomo model of the problem to be solved.
   
    Returns
    -------
    feasible_box : dict
        having the variables as keys and a list of corresponding lower and
        upper variable bounds as values.

    """
    
    model = call_model(0)
    
    feasible_box = {v.name: [v.lb, v.ub, str(v.domain)] for v in
                    model.component_objects(Var)}
    
    return feasible_box