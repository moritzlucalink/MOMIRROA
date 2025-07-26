#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 15:22:08 2025

@author: moritz
"""

from pyomo.environ import *

import copy as cp
import numpy as np
import time

from branch_box import *
from direct_image_box import *
from discarding_test import *
from feasible_point import *
from ideal_point import *
from init_box_list import *
from plot_enclosure import *
from plot_nondom import *
from update_lowerbounds import *
from update_lubs_nodefs import *
from update_nondom import *
from width_computation import *

def MOMIBB_direct(call_model, parameter, options, zu=[], zl=[]):
    
    # catch parameters
    m = parameter.m
    tol = parameter.tol
    maxiter = parameter.maxiter
    timeout = parameter.timeout
    gap_tolerance = 1e-6
    direct_solver = 'scip'
    
    dir_vec = np.ones(m)
    
    # check if plots should be shown
    try:
        show_plots = options.show_plots
    except:
        show_plots = False
    
    # check if image box is already given
    if len(zu) == 0:
        # compute image box
        zl, zu = direct_image_box(
            call_model,
            m,
            gap_tolerance,
            direct_solver)
    
    # initialize list of preimage space boxes with corresponding ideal points
    box_list = [[init_box_list(call_model, parameter), zl]]
    
    # initialize set of ideal points 
    A = [zl]
    
    # initialize set of potentially nondominated points
    N = []
    
    # initialize set of local upper bounds
    lubs = [zu]
    
    # compute initial width
    width, worst_llb, worst_lub = compute_width(
        A,
        lubs,
        dir_vec)
    
    # initialize time and iteration count
    it = 0
    total_time = 0
    start_time = time.time()
    
    minlp_count = 0
    
    while (len(box_list) > 0 and width > tol and it < maxiter and total_time < timeout):
        
        # print loop information
        print('\niteration # ', it, ' | current width:', round(width,3))
        print('current # of boxes:', len(box_list), ' | # of nondom points:',
              len(N))
        
        # catch index of box to branch on
        worst_box_idx = next(
            (i for i,(d,a) in enumerate(box_list) if np.array_equal(
                a,
                worst_llb
                )),
            None
        )
        
        # extract and catch box to branch on from active box list
        worst_box = box_list.pop(worst_box_idx)
        
        # branching rule
        new_box1, new_box2 = branch_box(worst_box[0])
        
        # compute first box ideal point
        ideal_point1 = compute_ideal_point(call_model,
                                          m,
                                          new_box1,
                                          timeout,
                                          direct_solver)
        
        # count minlp
        minlp_count += 1
        
        # check if new_box1 can be discarded
        if ideal_point1 is not None:
            keep, minlp_count = discarding_test(call_model,
                                                new_box1,
                                                ideal_point1,
                                                lubs,
                                                timeout,
                                                direct_solver,
                                                minlp_count)
        else:
            keep = False
            
        # process new_box1
        if keep:
            box_list.append([new_box1, ideal_point1])
            
            y = compute_feasible_point(call_model,
                               m,
                               new_box1,
                               ideal_point1,
                               timeout,
                               direct_solver)
            
            minlp_count += 1
            N, check = update_nondom(N, y)
            
            if check == 1:
                lubs = update_lubs(lubs, y)
        
        # compute second box ideal point
        ideal_point2 = compute_ideal_point(call_model,
                                          m,
                                          new_box2,
                                          timeout,
                                          direct_solver)
        
        # count minlp
        minlp_count += 1
        
        # check if new_box2 can be discarded
        if ideal_point2 is not None:
            keep, minlp_count = discarding_test(call_model,
                                                new_box2,
                                                ideal_point2,
                                                lubs,
                                                timeout,
                                                direct_solver,
                                                minlp_count)
        else:
            keep = False

        # process new_box2        
        if keep:
            box_list.append([new_box2, ideal_point2])
            
            y = compute_feasible_point(call_model,
                               m,
                               new_box2,
                               ideal_point2,
                               timeout,
                               direct_solver)
            
            minlp_count += 1
            N, check = update_nondom(N, y)
            
            if check == 1:
                lubs = update_lubs(lubs, y)
        
        # catch active ideal points
        A = [ideal_point for info, ideal_point in box_list]
        
        # compute width
        width, worst_llb, worst_lub = compute_width(
            A,
            lubs,
            dir_vec)
        
        # update iteration count and time
        it += 1
        current_time = time.time()
        total_time = current_time - start_time
    
    # get final list of lower bounds aka nondominated ideal points
    A = [ideal_point for info, ideal_point in box_list]
    L = [zu]
    for ideal_point in A:
        L, check = update_nondom(L, ideal_point)
    
    # print results
    print('finished after', it, 'iterations')
    print('finished after', total_time, 's')
    print('finished with', len(box_list), 'active boxes')
    print('computed', len(N), 'nondominated points')
    print('solved', minlp_count, 'MINLPs')
    
    # build return
    encl_dict = {}
    encl_dict['zu'] = zu
    encl_dict['zl'] = zl
    encl_dict['lubs'] = lubs
    encl_dict['llbs'] = L
    encl_dict['N'] = N
    encl_dict['total_time'] = total_time
    encl_dict['width'] = width
    encl_dict['MINLP_count'] = minlp_count
    encl_dict['box_list'] = box_list
    
    if show_plots:
        plot_nondom(N, m, total_time, it, tol, width)
        plot_enclosure(encl_dict, m, total_time, it, tol, width)
    
    return encl_dict, it
    