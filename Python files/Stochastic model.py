#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:42:48 2020

@author: isa
"""

#stochastic forms

#Br - resident duplication
#Bc - challenger duplication
#Dr - resident cell death
#Cr - challenger cell death


# dR = (Br - Dr)dt + (sqrt(Br + Dr))dW(t)
# dC = (Bc - Dc)dt + (sqrt(Bc + Dc))dW(t)

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

###############################################################################
#tau leaping

N = 3

@jit(nopython=True)
def tauNspecies(t, X, N, K, r, alpha, tau, batch_size=1, random_state=None):
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

#create params
N = 3
t = 100 # max time, starting from zero
tau = 0.01 # time step size
X = [750, 900, 1250] # initial population sizes
K = [10000,10000,10000] # carrying capacities
r = [1, 1,1] # growth rates
# Competition - must have '1' on the diagonal
alpha = array([[1, 1, 0.2], [0.5, 1, 0.34], [1,0.9, 1]])
ta, Xa = tauNspecies(t, X, N, K, r, alpha, tau)

    


def plottau(ta, Xa):    
    for index, pop in enumerate(Xa):
        #print(ta)
       # print(pop)
        plt.plot(ta, pop, label = 'Population '+str(index))
    plt.legend()

    
    
    plt.title('Stochastic System')
    plt.xlabel('Time')
    plt.ylabel('Population')
plottau(ta, Xa)


###############################################################################


#elfi


def popdraw(t, X, N, K, r, alpha, tau, value1, value2, batch_size=1, random_state=None):
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
            value1, value2 = (np.random.poisson(birth_rate * tau, 1) -
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
    return poisson.rvs(value1, value2, size=1000, size=(batch_size, 30), random_state=random_state)


def log_destack(final_pops):
    x = final_pops.flatten().reshape(1, -1)
    return(np.log(np.fmax(x, np.zeros(x.shape)) + 1))

alpha = elfi.Prior(ss.poisson, 0, 2)
t = elfi.Prior(ss.poisson, 0, 3)



y_obs = tauNspecies(t, X, N, K, r, alpha, tau)


vectorized_simulator = elfi.tools.vectorize(tauNspecies, [2, 3])


Y = elfi.Simulator(vectorized_simulator, alpha, t, observed=y_obs)

S = elfi.Summary(log_destack, Y)

d = elfi.Distance('euclidean', S)
