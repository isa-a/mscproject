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
X = [round(0.00017064*(10**5)), round(0.00096095*(10**5))]
K = [round(3.2082*(10**5)), round(2.145*(10**5))] # carrying capacities

#K = [round(1.79*(10**5)), round(8.25*(10**5))] # carrying capacities
r = [0.39181, 0.83005] # growth rates
Pops = np.array([[0.00017064,	0.0040678,	0.023875,	0.023706,	0.081055,	0.031941,	0.070163,	1.8389,	1.7067],
                              [0.00096095,	0.075931,	0.003776,	0.0020878	, 0.0019161,	0.00017784,	0.00086743,	0.0024855,	0.0025486]])
Pops = Pops*(10**5)
alpha = np.array([[1, 0.30301], [0.44315, 1]])

simtimes, simpops = tauNspecies(t, X, N, K, r, alpha, tau)

#len(simpops[1])
#simpops = np.asanyarray([simpops])
#simpops = simpops.reshape(2101,2)

#Pops = Pops.reshape(9, 2)

measurementTimes = np.array(([2,3,4,5,6,7,12,16,23]))

#measurementTimes = np.array(([0,1,2,3,4,5,10,14,21]))
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



