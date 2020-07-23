#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 16:52:19 2020

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
#from graphviz import Digraph

from SDEint import tauNspecies

N = 2
t = 100 # max time, starting from zero
tau = 0.01 # time step size
X = [750, 900] # initial population sizes
K = [10000,10000] # carrying capacities
r = [1, 1] # growth rates
# Competition - must have '1' on the diagonal
alpha = array([[1, 1.5], [0.5, 1]])
ta, Xa = tauNspecies(t, X, N, K, r, alpha, tau)

# measurement_times = np.linspace(0, 100, 10)
measurement_times = sorted([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
m_idx = 0
measurement_idx = []
for t_idx, t in enumerate(ta):
    if t >= measurement_times[m_idx]:
        measurement_idx.append(t_idx)
        m_idx += 1

if len(measurement_idx) < len(measurement_times):
    sys.stderr.write("Not all measurement times found\n")

Xa = np.array(Xa)
ta = np.array(ta)
Xa = Xa[:, measurement_idx]
ta = ta[measurement_idx]
