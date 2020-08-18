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
t = 21 # max time, starting from zero
tau = 0.01
X = [0.00015848*(10**11), 0.00017064*(10**11)]
K = [1.79*(10**11), 8.25*(10**11)] # carrying capacities
r = [0.39181*(10**11), 0.2297*(10**11)] # growth rates
Pops = np.array([[3.1713,	1.7697,	-0.007078,	1.2294,	-0.93122,	0.15738,	0.81653,	-0.010661,	0.067473], 
                             [0.17005,	-1.3495,	-0.048992, 	3.0277,	-1.5105,	-0.078494,	0.2635,	0.0036116,	-0.12046]])
scale = (10**11)
Pops = Pops*(10**11)
#len(Pops[1])

alpha = np.array([[1, 0.18179], [0.37922, 1]])

simtimes, simpops = tauNspecies(t, X, N, K, r, alpha, tau)
#len(simpops[1])
simpops = np.asanyarray([simpops])
simpops = simpops.reshape(2101,2)

def plotdata(simtimes, simpops):    
    for index, pop in enumerate(simpops):
        #print(ta)
       # print(pop)
        plt.plot(simtimes, pop, label = 'Population '+str(index))
    plt.legend()
    
    plt.title('Stochastic System')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend(['Population 1', 'Population 2'])

    plt.ylabel('Population')
plotdata(simtimes, simpops)

measurementTimes = np.array(([0,1,2,3,4,5,10,14,21]))
measurementTimes = measurementTimes*scale
measurement_idx = np.array([int(x / tau) for x in measurementTimes])


