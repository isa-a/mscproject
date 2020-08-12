#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 19:45:05 2020

@author: isa
"""

import sys, os
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numpy import *
import numpy as np
from numba import jit
import elfi
import scipy.stats as ss
from scipy.stats import poisson

N = 2
t = 21 # max time, starting from zero
K = [1000,1000] # carrying capacities
r = [1.5, 1.5] # growth rates
Pops = np.array([[3.1713,	1.7697,	-0.007078,	1.2294,	-0.93122,	0.15738,	0.81653,	-0.010661,	0.067473], [0.17005,	-1.3495,	-0.048992, 	3.0277,	-1.5105,	-0.078494,	0.2635,	0.0036116,	-0.12046]])
#len(Pops[1])
measurementTimes = (([0,1,2,3,4,5,10,14,21]))








