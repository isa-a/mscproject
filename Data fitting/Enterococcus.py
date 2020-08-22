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

from SDEint import tauNspecies

N = 2
t = 23# max time, starting from zero
tau = 0.01
X = [round(0.00017064*(10**5)), round(0.00029079*(10**5))]
K = [round(3.2082*(10**5)), round(0.15152*(10**5))] # carrying capacities

#K = [round(1.79*(10**5)), round(8.25*(10**5))] # carrying capacities
r = [0.39181, 0.29075] # growth rates
Pops = np.array([[0.00017064,	0.0040678,	0.023875,	0.023706,	0.081055,	0.031941,	0.070163,	1.8389,	1.7067],
                              [0.00029079,	0.0081357,	0.0015592,	0.0029692	, 0.013796	, 0.00084614,	0.0072227,	0.011825,	0.036377]])
Pops = Pops*(10**5)
alpha = np.array([[1, 0.01436], [0.11124, 1]])

simtimes, simpops = tauNspecies(t, X, N, K, r, alpha, tau)

#len(simpops[1])
#simpops = np.asanyarray([simpops])
#simpops = simpops.reshape(2101,2)

#Pops = Pops.reshape(9, 2)

measurementTimes = np.array(([2,3,4,5,6,7,12,16,23]))
measurement_idx = np.array([int(x / tau) for x in measurementTimes])



def plottau(measurementTimes, Pops):    
    for index, pop in enumerate(Pops):
        #print(ta)
       # print(pop)
        plt.plot(measurementTimes, pop, label = 'Population '+str(index))
    plt.legend()

    
    
    plt.title('D')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend(['Population 1', 'Population 2'])

    plt.ylabel('Population')
plottau(simtimes, simpops)



