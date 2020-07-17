#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 16:23:06 2020

@author: isa
"""
#import all modules required
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

#import SDE sim from seperate file and observed data from seperate file
from SDEint import tauNspecies
from obs import ta,Xa



alphaprior1 = elfi.Prior(ss.uniform, 0, 2)
alphaprior2 = elfi.Prior(ss.uniform, 1,3)

observeddata = ta, Xa

#wrap create
def tauWrapper(t, X, N, K, r, alphavector, tau, batch_size=1, random_state=None):
    final_pop = []

    times, pops = tauNspecies(t, X, N, K, r, alphavector, tau)

    final_pop.append((pops[-1,0], pops[-1,1]))

    return(np.asarray(times, pops))
#tauWrapper(100, [780, 300], 2, [1000,1000], [1,1], array([[1, alphaprior1], [alphaprior2, 1]]), 0.01)



def logdestack(final_pop):
    x = final_pop.flatten().reshape(1, -1)
    return(np.log(np.fmax(x, np.zeros(x.shape)) + 1))
    

vectorized_sim = elfi.tools.vectorize(tauWrapper, [1,2,3,4,5])

Y = elfi.Simulator(vectorized_sim, alphaprior1, alphaprior2, observed=observeddata)

S = elfi.Summary(logdestack, Y)
d = elfi.Distance('euclidean', S)
log_d = elfi.Operation(np.log, d)


bolfi = elfi.BOLFI(log_d, batch_size=1, initial_evidence=20, update_interval=10, bounds={'alphaprior1':(0, 3), 'alphaprior2':(1, 5)}, acq_noise_var=[0.01, 0.01], seed=1)
    
post = bolfi.fit(n_evidence=20)








    
    
    
    
    
    
    
    
    
    