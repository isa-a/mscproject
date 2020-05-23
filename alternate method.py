#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 06:29:00 2020

@author: isa
"""
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numpy import *

#creating function with the LV equations within it, first listing all parameters
def lotkavolterra(t, X, K, rR, rC, rC2, alphaCR, alphaRC, alphaC2R, alphaC2C):
     R, C, C2 = X
     return [(rR*R)*((K - R- (alphaCR*C)) / K), (rC*C)*((K - C - (alphaRC*R)) / K), (rC2*C2)*((K - C2 - (alphaC2R*C2) - (alphaC2C*C2)) / K)]



solve = solve_ivp(lotkavolterra, [0, 100], [50, 200, 350], args=(1000, 1, 1, 1, 0.8, 0.8, 0.8, 0.46), dense_output=True) #passing through the values into the scipy solver



t = linspace(0, 100, 100) #start and stop times and the no. of samples to generate
X = solve.sol(t)
plt.plot(t, X.T)
plt.legend(['R', 'C','C2'])
plt.title('Lotka-Volterra System')
plt.show()


