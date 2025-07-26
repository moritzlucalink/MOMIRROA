#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:44:29 2024

@author: moritz
"""

import itertools as iter
import numpy as np
import os
import sys

sys.path.append('../MOMIRROA_methods')
sys.path.append('../MOMIBB_methods')

from MOMIRROA import *
from MOMIBB_direct import *


tols = [0.1]#, 0.1, 0.05, 0.01]
methods = ['MOMIRROA']#, 'MOMIBB']
delta_factors = [0.5, 0.1]
cons_tols = [1e-4]#, 1e-6]#, 1e-4, 1e-6]
nlp_feas = [0]#,1]
momirroa_configs = ['AD', 'AD-BT', 'UN']

k=2

for k, l, tol, method in iter.product([6],
                                      [6],
                                      tols,
                                      methods):

    print('k=',k,'l=',l,'method:',method,'tol=',tol)

    imported_file = __import__('TI20_k'+str(k)+'_l'+str(l))
    build_model = imported_file.build_model

    options = imported_file.options
    parameter = imported_file.parameter

    parameter.tol = tol

    options.show_plots = False
    options.solve_direct = False
    parameter.maxiter = 20000

    if method == 'MOMIBB':
        encl_dict, it = MOMIBB_direct(build_model, parameter, options)

        path = '../Numerical_Results/TI20_k'+str(k)+'_l'+str(l)+\
            '/tol'+str(tol)+'/MOMIBB/'

        try:
            os.makedirs(path)
        except:
            None

        f = open(path+'summary.txt', 'w')
        f.write('summary of analysis\n')
        f.write('total time: ' + str(encl_dict['total_time']))
        f.write('\niterations: ' + str(it))
        f.write('\nwidth: ' + str(encl_dict['width']) + ' tol: ' + str(tol))
        f.write('\n# of MINLPs:' + str(encl_dict['MINLP_count']))
        f.write('\n# of nondominated points:' + str(len(encl_dict['N'])))
        f.write('\n# of active boxes:' + str(len(encl_dict['box_list'])))

        f.close()

    elif method == 'MOMIRROA':
        for delta_factor, cons_tol, feasibility, config in iter.product(
                delta_factors,
                cons_tols,
                nlp_feas,
                momirroa_configs):

            parameter.factor_delta = delta_factor * tol

            # configure solver
            options.constraint_tolerance = cons_tol
            options.nlp_feasibility_only = feasibility

            if config == 'UN':
                options.adaptive_refinement = False
                options.bound_tightening = 1e20

            elif config == 'AD':
                options.adaptive_refinement = True
                options.bound_tightening = 1e20

            elif config == 'AD-BT':
                options.adaptive_refinement = True
                options.bound_tightening = 3

            encl_dict, it = MOMIRROA(build_model, parameter, options)

            path = '../Numerical_Results/TI20_k'+str(k)+'_l'+str(l)+\
                '/tol'+str(tol)+'/MOMIRROA/deltafac'+str(delta_factor)+\
                '/cons_tol'+str(cons_tol)+'/NLPfeas'+str(feasibility)+\
                '/'+str(config)+'/'

            try:
                os.makedirs(path)
            except:
                None

            f = open(path+'summary.txt', 'w')
            f.write('summary of analysis\n')
            f.write('total time: ' + str(encl_dict['total_time']))
            f.write('\niterations: ' + str(it))
            f.write('\nwidth: ' + str(encl_dict['width']) + 'tol: ' + str(tol))
            f.write('\n# of nondominated points:' + str(len(encl_dict['N'])))
            f.write('\n\nmaximal # of preimage set boxes: ' + str(max([encl_dict['analysis'][str(i)]['maxpreimageboxes'] for i in np.arange(0,it)])))
            f.write('\nAVG # of preimage set boxes: ' + str(sum(encl_dict['analysis'][str(i)]['preimageboxcounter'] for i in np.arange(0,it))/sum(encl_dict['analysis'][str(i)]['relaxedproblemcounter'] for i in np.arange(0,it))))
            f.write('\n# of MILPs: ' + str(sum(encl_dict['analysis'][str(i)]['relaxedproblemcounter'] for i in np.arange(0,it))))
            f.write('\nAVG time for MILPs: ' + str(sum(encl_dict['analysis'][str(i)]['relaxed_solution_time'] for i in np.arange(0,it))/sum(encl_dict['analysis'][str(i)]['relaxedproblemcounter'] for i in np.arange(0,it))))
            f.write('\n# of NLPs: ' + str(sum(encl_dict['analysis'][str(i)]['problemcounter'] for i in np.arange(0,it))))
            f.write('\nAVG time for NLPs: ' + str(sum(encl_dict['analysis'][str(i)]['solution_time'] for i in np.arange(0,it))/sum(encl_dict['analysis'][str(i)]['problemcounter'] for i in np.arange(0,it))))
            f.write('\ntime spent for bound tightening: ' + str(sum(encl_dict['analysis'][str(i)]['time bound tightening'] for i in np.arange(0,it))))

            f.write('\n# of OBBT MILPs: ' + str(sum(encl_dict['analysis'][str(i)]['# of OBBT MILPs'] for i in np.arange(0,it))))
            f.write('\nshare of search zone improvement by feas-dec (total): ' + str(sum(encl_dict['analysis'][str(i)]['# of considered feasible'] for i in np.arange(0,it))/sum(encl_dict['analysis'][str(i)]['# of search zones'] for i in np.arange(0,it))))

            f.write('\n\niteration information:')
            f.write('\n# of PSB per iteration:')
            for i in np.arange(0,it):
                f.write('\nMax # PSB in iteration' + str(i) +': ' + str(encl_dict['analysis'][str(i)]['maxpreimageboxes']))

            f.write('\n\nAVG # PSB per iteration:')
            for i in np.arange(0,it):
                f.write('\nAVG # PSB in iteration' + str(i) + ': ' +str(encl_dict['analysis'][str(i)]['preimageboxcounter']/encl_dict['analysis'][str(i)]['relaxedproblemcounter']))

            f.write('\n\n# of MILPs per iteration:')
            for i in np.arange(0,it):
                f.write('\n# of MILPs in iteration' + str(i) + ': ' + str(encl_dict['analysis'][str(i)]['relaxedproblemcounter']))

            f.write('\n\nAVG time for MILPs per iteration:')
            for i in np.arange(0,it):
                f.write('\nAVG time for MILPs in iteration' + str(i) + ': ' + str(encl_dict['analysis'][str(i)]['relaxed_solution_time']/encl_dict['analysis'][str(i)]['relaxedproblemcounter']))

            f.write('\n\n# of NLPs per iteration:')
            for i in np.arange(0,it):
                f.write('\n# of NLPs in iteration' + str(i) + ': ' + str(encl_dict['analysis'][str(i)]['problemcounter']))

            f.write('\n\nAVG time for NLPs per iteration:')
            for i in np.arange(0,it):
                try:
                    f.write('\nAVG time for NLPs in iteration' + str(i) + ': ' + str(encl_dict['analysis'][str(i)]['solution_time']/encl_dict['analysis'][str(i)]['problemcounter']))
                except:
                    None

            f.write('\n\n')
            for i in np.arange(0,it):
                f.write('\nshare of search zone improvement by feas-dec in' + str(i) + '-th iteration:' + str(encl_dict['analysis'][str(i)]['# of considered feasible']/encl_dict['analysis'][str(i)]['# of search zones']))
            f.write('\nshare of enforced search zone improvement (total):' + str(sum(encl_dict['analysis'][str(i)]['# of enforced feasibility check'] for i in np.arange(0,it))/sum(encl_dict['analysis'][str(i)]['# of search zones'] for i in np.arange(0,it))))
            for i in np.arange(0,it):
                f.write('\nshare of enforced search zone improvement in' + str(i) + '-th iteration:' + str(encl_dict['analysis'][str(i)]['# of enforced feasibility check']/encl_dict['analysis'][str(i)]['# of search zones']))

            f.close()










    # if not direct_solve:
    #     for cons_tol, feasibility, adaptive, BT in iter.product(cons_tols,
    #                                                           boolean,
    #                                                           boolean,
    #                                                           BT_vals):


    #         options.constraint_tolerance = cons_tol
    #         options.nlp_feasibility_only = feasibility
    #         options.adaptive_refinement = adaptive
    #         options.bound_tightening = BT

    #         encl_dict, it = compute_enclosure(build_model, parameter, options)

    #         path= '/home/moritz/Nextcloud/PhD/Dissertation/Numerics/tol'+str(tol)+\
    #                     '/deltafac'+str(delta_factor)+'/TI20_k'+str(k)+'_l'+str(l)+'/2Stage/cons_tol'+str(cons_tol)+\
    #                         '/NLPfeas'+str(feasibility)+'/adaptive'+str(adaptive)+'/BT'+\
    #                     str(BT)+'/'

    #         try:
    #             os.makedirs(path)
    #         except:
    #             None

    #         f = open(path+'summary.txt', 'w')
    #         f.write('summary of analysis\n')
    #         f.write('total time: ' + str(encl_dict['total_time']))
    #         f.write('\n iterations: ' + str(it))
    #         f.write('\n width: ' + str(encl_dict['width']) + 'tol: ' + str(tol))
    #         f.write('\n maximal # of preimage set boxes: ' + str(max([encl_dict['analysis'][str(i)]['maxpreimageboxes'] for i in np.arange(0,it)])))
    #         f.write('\n AVG # of preimage set boxes: ' + str(sum(encl_dict['analysis'][str(i)]['preimageboxcounter'] for i in np.arange(0,it))/sum(encl_dict['analysis'][str(i)]['relaxedproblemcounter'] for i in np.arange(0,it))))
    #         f.write('\n # of MILPs: ' + str(sum(encl_dict['analysis'][str(i)]['relaxedproblemcounter'] for i in np.arange(0,it))))
    #         f.write('\nAVG time for MILPs: ' + str(sum(encl_dict['analysis'][str(i)]['relaxed_solution_time'] for i in np.arange(0,it))/sum(encl_dict['analysis'][str(i)]['relaxedproblemcounter'] for i in np.arange(0,it))))
    #         f.write('\n # of NLPs: ' + str(sum(encl_dict['analysis'][str(i)]['problemcounter'] for i in np.arange(0,it))))
    #         f.write('\n AVG time for NLPs: ' + str(sum(encl_dict['analysis'][str(i)]['solution_time'] for i in np.arange(0,it))/sum(encl_dict['analysis'][str(i)]['problemcounter'] for i in np.arange(0,it))))
    #         f.write('\n time spent for bound tightening: ' + str(sum(encl_dict['analysis'][str(i)]['time bound tightening'] for i in np.arange(0,it))))

    #         f.write('\n # of OBBT MILPs: ' + str(sum(encl_dict['analysis'][str(i)]['# of OBBT MILPs'] for i in np.arange(0,it))))
    #         f.write('\n share of search zone improvement by feas-dec (total): ' + str(sum(encl_dict['analysis'][str(i)]['# of considered feasible'] for i in np.arange(0,it))/sum(encl_dict['analysis'][str(i)]['# of search zones'] for i in np.arange(0,it))))

    #         f.write('\n\n iteration information:')
    #         f.write('\n # of PSB per iteration:')
    #         for i in np.arange(0,it):
    #             f.write('\n Max # PSB in iteration' + str(i) +': ' + str(encl_dict['analysis'][str(i)]['maxpreimageboxes']))

    #         f.write('\n\n AVG # PSB per iteration:')
    #         for i in np.arange(0,it):
    #             f.write('\n AVG # PSB in iteration' + str(i) + ': ' +str(encl_dict['analysis'][str(i)]['preimageboxcounter']/encl_dict['analysis'][str(i)]['relaxedproblemcounter']))

    #         f.write('\n\n # of MILPs per iteration:')
    #         for i in np.arange(0,it):
    #             f.write('\n # of MILPs in iteration' + str(i) + ': ' + str(encl_dict['analysis'][str(i)]['relaxedproblemcounter']))

    #         f.write('\n\n AVG time for MILPs per iteration:')
    #         for i in np.arange(0,it):
    #             f.write('\n AVG time for MILPs in iteration' + str(i) + ': ' + str(encl_dict['analysis'][str(i)]['relaxed_solution_time']/encl_dict['analysis'][str(i)]['relaxedproblemcounter']))

    #         f.write('\n\n # of NLPs per iteration:')
    #         for i in np.arange(0,it):
    #             f.write('\n # of NLPs in iteration' + str(i) + ': ' + str(encl_dict['analysis'][str(i)]['problemcounter']))

    #         f.write('\n\n AVG time for NLPs per iteration:')
    #         for i in np.arange(0,it):
    #             try:
    #                 f.write('\n AVG time for NLPs in iteration' + str(i) + ': ' + str(encl_dict['analysis'][str(i)]['solution_time']/encl_dict['analysis'][str(i)]['problemcounter']))
    #             except:
    #                 None

    #         f.write('\n\n')
    #         for i in np.arange(0,it):
    #             f.write('\nshare of search zone improvement by feas-dec in' + str(i) + '-th iteration:' + str(encl_dict['analysis'][str(i)]['# of considered feasible']/encl_dict['analysis'][str(i)]['# of search zones']))
    #         f.write('\n share of enforced search zone improvement (total):' + str(sum(encl_dict['analysis'][str(i)]['# of enforced feasibility check'] for i in np.arange(0,it))/sum(encl_dict['analysis'][str(i)]['# of search zones'] for i in np.arange(0,it))))
    #         for i in np.arange(0,it):
    #             f.write('\n share of enforced search zone improvement in' + str(i) + '-th iteration:' + str(encl_dict['analysis'][str(i)]['# of enforced feasibility check']/encl_dict['analysis'][str(i)]['# of search zones']))

    #         f.close()

    # else:
    #     encl_dict, it = compute_enclosure(build_model, parameter, options)

    #     path = '~/Nextcloud/PhD/Dissertation/Numerics/tol'+str(tol)+\
    #     '/deltafac'+str(delta_factor)+'/TI20_k'+str(k)+'_l'+str(l)+'/direct/'

    #     try:
    #         os.makedirs(path)
    #     except:
    #         None

    #     f = open(path+'summary.txt', 'w')
    #     f.write('summary of analysis\n')
    #     f.write('total time: ' + str(encl_dict['total_time']))
    #     f.write('\n iterations: ' + str(it))
    #     f.write('\n width: ' + str(encl_dict['width']) + ' tol: ' + str(tol))
    #     f.write('\n # of MINLPs:' + str(sum(encl_dict['analysis'][str(i)]['problemcounter'] for i in np.arange(0,it))))
    #     f.write('\n AVG time for MINLPs: ' + str(sum(encl_dict['analysis'][str(i)]['solution_time'] for i in np.arange(0,it))/sum(encl_dict['analysis'][str(i)]['problemcounter'] for i in np.arange(0,it))))

    #     f.write('\n\n # of MINLPs per iteration:')
    #     for i in np.arange(0,it):
    #         f.write('\n # of MINLPs in iteration ' +str(i) +': ' + str(encl_dict['analysis'][str(i)]['problemcounter']))

    #     f.write('\n\n AVG time for MINLPs per iteration:')
    #     for i in np.arange(0,it):
    #         f.write('\n AVG time for MINLPs in iteration ' +str(i) +': ' + str(encl_dict['analysis'][str(i)]['solution_time']/encl_dict['analysis'][str(i)]['problemcounter']))

    #     f.close()
