#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 16:33:41 2020

@author: isa
"""
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numpy import *
import numpy as np
import gillespy2
from sdeint import *
from numba import jit
import elfi
import scipy.stats as ss
from scipy.stats import poisson
from graphviz import Digraph



@jit(nopython=True)
def tauNspecies(t, X, N, K, r, alpha, tau):
    # Time steps
    ta = [0]

    # Set up initial condition for each population
    # List of lists - each population is a list item
    # Each population list Xa[i][j] is the population i size at ta[j]
    Xa = list()
    for pop in X:
        Xa.append([pop])

    # Xa = [[50], [200], [350]]
    # Xa[0] = [50]
    # Xa[0][0] = 50
    t_current = 0
    step_idx = 0
    while (t_current < t):

        for pop_i in range(N):
            birth_rate = r[pop_i] * Xa[pop_i][step_idx]
            death_rate = 0
            for pop_j in range(N):
                death_rate += (r[pop_i] * Xa[pop_i][step_idx]) * (alpha[pop_j, pop_i] * Xa[pop_j][step_idx] / K[pop_i])
          #  print(birth_rate)
            #print(death_rate)
            population_change = (np.random.poisson(birth_rate * tau, 1) -
                                 np.random.poisson(death_rate * tau, 1))[0]
           # print('pop change' + str(population_change))
            new_size = Xa[pop_i][step_idx] + population_change
            if new_size >= 0:
                Xa[pop_i].append(new_size)
            else:
                Xa[pop_i].append(0)

        step_idx += 1
        t_current += tau
        ta.append(t_current)

    #ta = [t + T_init for t in ta]
    return(ta, Xa)
