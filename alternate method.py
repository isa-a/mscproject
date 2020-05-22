#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 06:29:00 2020

@author: isa
"""
from scipy.integrate import solve_ivp
 import matplotlib.pyplot as plt

#creating function with the LV equations within it, first listing all parameters
def lotkavolterra(t, X, K, R, C, rR, rC, alphaCR, alphaRC):
     dR, dC = X
     return [(rR*R)*((K - R- (alphaCR*C)) / K), (rC*C)*((K - C - (alphaRC*R)) / K)]



solve = solve_ivp(lotkavolterra, [0, 15], [0.001, 100], args=(1000, 1000, 1000, 1, 1, 0.8, 0.8), dense_output=True) #passing through the values into the scipy solver



t = linspace(0, 15, 50) #start and stop times and the no. of samples to generate
X = solve.sol(t)
plt.plot(t, X.T)
plt.legend(['R', 'C'])
plt.title('Lotka-Volterra System')
plt.show()


