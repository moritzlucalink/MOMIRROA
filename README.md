# Multi-objective-optimization
This repository contains an algorithm for computing an enclosure of the nondominated set of multi-objective mixed-integer nonlinear optimization problems.

README file for codes related to 
	Eichfelder, G. and Link, M. and Volkwein, S. and Warnow, L.
	An adaptive relaxation-refinement scheme for multi-objective mixed-integer nonconvex optimization
	available at: https://optimization-online.org/?p=27337
	
- it is recommended to run the files in an virtual environment

- the file "requirements.txt" contains the specifications for a suitable VE

- the directory "~/mains/" contains all main files corresponding to the problems mentioned in the above paper

- the file "TI22.py" contains problem P1 from the paper
- the files "TI20_kX_lX.py" contain the problem P2 with the respective configuration of k and l

- to solve a problem, one simply runs 		python TI22.py		with the VE activated

- the directory "~/MOMIRROA_methods/" contains all files corresponding to the algorithm and its variants presented in the paper in question

- all code files contain a brief description of the methods written in the respective file

- we briefly explain the structure of the "~/MOMIRROA_methods/" directory:
	
	- the main file is named "MOMIRROA.py"
	- it contains the main methods (similar to Algorithm 1 in the paper)
	- it starts with catching the parameters of the algorithm (image space dimension, width tolerance, maxiter, timeout, and delta)
	- it continues with checking the specifications of the algorithm
	
	- the main specifications which are to be set are:
		- options.nlp_feasibility_only = True/False 	deciding if the NLPs originating from fixing the integer assignments of the MINLP should be solved to feasibility only (True) or to global optimality (False)
		
		- options.adaptive_refinement = True/False	deciding if the adaptive (True) or uniform (False) refinement scheme should be applied
		
		- options.bound_tightening = integer/comment	indicating after how many upper bound updates the OBBT procedure should be applied; if no OBBT is thought to be applied: comment this line or set it to 1e20

    		- options.constraint_tolerance = float		determining the theta value frm the paper, i.e., which constraint violation is considered to be acceptable
		
	- the rest of the specifications are not considered in the paper in question; namely these are:
		- options.solve_direct = True/False		only the False option is used in the paper
		- options.gap_tolerance = float			only relevant in the above True option
		
		- options.show_plots = True/False		deciding if summary and plots are to be printed
		
		- options.soft_utopian_check = True/False	only the True option is used in this paper

    		- options.method = string			MOMIRROA or MOMIBB (see below for explanation)
		
	
- we briefly explain the structure of the main problem files:

	- it starts with the problem names 
	- afterwards a function for building the pyomo model is defined
	- the algorithm parameters and options (for both see above) are declared
	- the method "methods/MOMIRROA.py" is called
	
- for testing, e.g., the instance P2 with k=8 and l=2 with the adaptive refinement scheme without OBBT, and NLPs solved only to feasibility from the paper in question, one has to do the following:

	- open the file "~/mains/TI20_k8_l2.py"
	- set "options.nlp_feasibility_only = True"
	- set "options.adaptive_refinement = True"
	- comment the "options.bound_tightening"-line
	- run 			python TI20_k8_l2.py


## MOMIBB	
- there is also another algorithm, called MOMIBB, from the following paper
  		Eichfelder, G. and Stein, O. and Warnow, L.
  		A Solver for Multiobjective Mixed-Integer Convex and Nonconvex Optimization
  		Journal of Optimization Theory and Applications, 203, p. 1736-1766. 2023.
  		available at: https://link.springer.com/article/10.1007/s10957-023-02285-2

- the direct variant of MOMIBB is available as benchmark comparison in this repository
- one can choose to use it via selecting 'MOMIBB' for options.method


## Numerical Results
there is also a directory containing the numerical results underlying the analysis on the above paper. the corresponding experiments where run on a machine with Intel Core i7-8565U processor and 32GB of RAM. it futher used the SCIP Optimization Suite as well as the Gurobi MILP Solver.
the results are made available in a directory tree structure with single summary files as nodes but also via a single csv table containing all numbers.
