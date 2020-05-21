# -*- coding: utf-8 -*-
"""
Created on Sat May 16 17:06:41 2020

@author: isa
"""
from numpy import *

#K - carrying capacity
K =1000
#R - resident population size
R =100
#C - challenger population size
C =100
#rR - resident growth rate
rR = 1
#rC - challenger growth rate
rC = 1
#alphaCR - competition terms
alphaCR =0.8
#alphaRC - competition terms
alphaRC = 0.8

# model parameters
dt = 0.001; max_time = 100

# initial time
t = 0

# empty lists in which to store time and populations
t_list = []; R_list = []; C_list = []

# initialize lists
t_list.append(t); R_list.append(R); C_list.append(C)

while t < max_time:
    # calc new values for t, R, C
    # the equations below have simply been rearrnaged and intregrated once so they can be computed by python
    t = t + dt
    R = R + (rR*R)*(K - R - (alphaCR*C) / K)
    C = C + (rC*C)*((K - C - (alphaRC*R)) / K)

    # store new values in lists
    t_list.append(t)
    R_list.append(R)
    C_list.append(C)

# Plot the results    
p = plt.plot(t_list, R_list, 'r', t_list, C_list, 'g', linewidth = 2)










  #X = [R, C]
def dX_dt(X, t=0):
    return array([(rR*X[0])*((K - X[0] - (alphaCR*X[1])) / K), (rC*X[1])*((K - X[1] - (alphaRC*X[0])) / K)])
