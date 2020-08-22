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
from numba import jit
import elfi
import scipy.stats as ss
from scipy.stats import poisson
import time
import sys, os
#from graphviz import Digraph

#import SDE sim from seperate file and observed data from seperate file
from SDEint import tauNspecies
#from Akkermansia import N, t, tau, X, K, r, Pops, measurement_idx
#from Coprobacillus import N, t, tau, X, K, r, Pops, measurement_idx
from Enterococcus import N, t, tau, X, K, r, Pops, measurement_idx


starttime= time.time()
observeddata = Pops
#observeddata = np.loadtxt('testfile.txt')

#wrap create
def tauWrapper(t, X, N, K, r, alpha1, alpha2, tau, measurementTimes,
                batch_size=1, random_state=None):

    alpha = np.array([[1, alpha1], [alpha2, 1]])
    simtimes, simpops = tauNspecies(t, X, N, K, r, alpha, tau)

    measured_pop = np.array(simpops)[:, measurementTimes]
    return(measured_pop)
#tauWrapper(100, [780, 300], 2, [1000,1000], [1,1], array([[1, alphaprior1], [alphaprior2, 1]]), 0.01)


def logdestack(final_pop): #reshaping the output
    x = final_pop.flatten().reshape(1, -1) #flatten turns it into one dimension (1 column) and then reshape makes that into a row so it can be used later on
    return(np.log(np.fmax(x, np.zeros(x.shape)) + 1)) #taking logs and adding 1 as log0 not possible. fmax ensures any possible negative values are turned into 0

alphaprior1 = elfi.Prior(ss.uniform, 0, 5)
alphaprior2 = elfi.Prior(ss.uniform, 0, 5)

vectorized_sim = elfi.tools.vectorize(tauWrapper, [0, 1, 2, 3, 4, 7, 8])

simulator_results = elfi.Simulator(vectorized_sim, t, X, N, K, r, alphaprior1, alphaprior2,
                    tau, measurement_idx,
                    observed=observeddata)

summary = elfi.Summary(logdestack, simulator_results)
d = elfi.Distance('euclidean', summary)
#log_d = elfi.Operation(np.log, d)


bolfi = elfi.BOLFI(d, batch_size=1, initial_evidence=100, update_interval=10,
                    bounds={'alphaprior1':(0, 5), 'alphaprior2':(0, 5)},
                    acq_noise_var=[0.01, 0.01], seed=1)

post = bolfi.fit(n_evidence=500)

#bolfi.plot_state()
#plt.savefig("bolfi_state.pdf")
#plt.close()

bolfi.plot_discrepancy()
plt.savefig("bolfi_discrepancy.pdf")
plt.close()

#post.plot(logpdf=True)

#print(bolfi.target_model)


# sample from BOLFI posterior
sys.stderr.write("Sampling from BOLFI posterior\n")
result_BOLFI = bolfi.sample(100000, algorithm="metropolis", sigma_proposals=np.array([0.05, 0.05])) #rhat - measure of similarity between 4 chains (attempts at sampling posterior). want it to be close to 1
#effective sample size - want it to be around 200

print(result_BOLFI)
np.savetxt("samples.txt", result_BOLFI.samples_array)
print(np.mean(result_BOLFI.samples_array, axis = 0))
print(np.quantile(result_BOLFI.samples_array, q=[0.025, 0.975], axis = 0))


result_BOLFI.plot_traces()
plt.savefig("posterior_traces.pdf")
plt.close()

result_BOLFI.plot_pairs()
plt.savefig("pair_traces.pdf")
plt.close()

result_BOLFI.plot_marginals()
plt.savefig("posterior_marginals.pdf")
endtime= time.time()
print("time elapsed:", endtime - starttime)

