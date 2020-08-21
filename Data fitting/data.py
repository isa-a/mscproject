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
t = 21# max time, starting from zero
tau = 0.01
X = [0.00015848*(10**5), 0.00017064*(10**5)]
K = [1.79*(10**5), 8.25*(10**5)] # carrying capacities
r = [0.39181*(10**5), 0.2297*(10**5)] # growth rates
Pops = np.array([[0.00015848, 0.00018786,	0.0000487,	0.0000464,	0.00095807,	0.00021153,	0.00014287,	0.0004099	, 0.00042039], 
                            [0.00017064,	0.0040678,	0.023875,	0.023706,	0.081055,	0.031941,	0.070163,	1.8389,	1.7067]])
Pops = Pops*(10**5)
alpha = np.array([[1, 0.18179], [0.37922, 1]])

simtimes, simpops = tauNspecies(t, X, N, K, r, alpha, tau)
#len(simpops[1])
#simpops = np.asanyarray([simpops])
#simpops = simpops.reshape(2101,2)

Pops = Pops.reshape(9, 2)


measurementTimes = np.array(([0,1,2,3,4,5,10,14,21]))
measurementTimes = measurementTimes*(10**5)
measurement_idx = np.array([int(x / tau) for x in measurementTimes])



def plottau(measurementTimes, Pops):    
    for index, pop in enumerate(Pops):
        #print(ta)
       # print(pop)
        plt.plot(measurementTimes, Pops, label = 'Population '+str(index))
    plt.legend()

    
    
    plt.title('D')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend(['Population 1', 'Population 2'])

    plt.ylabel('Population')
plottau(measurementTimes, Pops)






