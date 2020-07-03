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

###############################################################################
#gillespie



@jit(nopython=True)#increase speed of function
def gillespy(Tinit, Tmax, R, C, K, rR, rC, alphaRC, alphaCR): #function with args
    ta = []
    Ra = []
    Ca = [] #empty lists to append to

    t = 0
    R = R
    C = C 
    
    while (t < Tmax - Tinit): #time step to integrate over
        ta.append(t)
        Ra.append(R)
        Ca.append(C)#adding to lists

        Br = rR * R
        Bc = rC * C
        Dr = rR/K * R * (R + alphaRC * C)
        Dc = rC/K * C * (C + alphaCR * R) #defining probabilities 

        R_sum = Br + Bc + Dr + Dc
        if (R_sum == 0): #if the population size is zero nothing will happen/can be done, so we must ensure the system ends if it is somehow zero
            break
        value1 = random.random()#generate random time 
        t += -log(value1)/R_sum

        value2 = random.random()#generate random population
        if (value2 < Br/R_sum):
            R += 1
        elif (value2 > Br/R_sum and value2 < (Br + Dr)/R_sum):
            R -= 1
        elif (value2 > (Br + Dr)/R_sum and value2 < (Br + Dr + Bc)/R_sum):
            C += 1
        else:
            C -= 1

    ta = [t + Tinit for t in ta]
    return(ta, Ra, Ca)


#timestep = [0,100]
#popsizes = [100,500]
#carryingcap = [1000]
#growthterms = [1.032,1.032]
#alphaterms = [1.5,1.3]    




def plotgillespie():
    results = gillespy(0,100,10,10,500,1.032,1.032,1,1)#storing the model
    
    
    pops = np.transpose(np.array([results[1], results[2]]))#writing results as a matrix
    
    plt.plot(results[0], pops)#plotting
    plt.ylim(0,(max(results[2])))#adjusting axis
    plt.title('Stochastic System')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend(['R', 'C'])

    #np.array([results[1], results[2]]).shape
plotgillespie()

###############################################################################
#tau leaping

N = 3

#@jit(nopython=True)
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
                death_rate += alpha[pop_i, pop_j] * Xa[pop_j][step_idx] / K[pop_i]
            population_change = (np.random.poisson(birth_rate * tau, 1) -
                                 np.random.poisson(death_rate * tau, 1))[0]
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
X = [50, 200, 350] # initial population sizes
K = [1000, 500, 430] # carrying capacities
r = [1, 1, 1] # growth rates
# Competition - must have '1' on the diagonal
alpha = array([[1, 0.5, 0.2], [0.3, 1, 0.4], [1.2, 0, 1]])
ta, Xa = tauNspecies(t, X, N, K, r, alpha, tau)
plt.plot(ta, Xa)

np.array([Xa[2], Xa[2]]).shape

def plottau():
    results = tauNspecies(t, X, N, K, r, alpha, tau)
    #results = ta, Xa    
   # pops = np.transpose(np.array([ta[1], ta[2]]))#writing results as a matrix
    pops = (np.array([ta[1], ta[2]]))#writing results as a matrix
    
    plt.plot(results[1], pops)#plotting
    plt.ylim(0,(max(results[2])))#adjusting axis
    plt.title('Stochastic System')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend(['R', 'C'])
    
plottau()

