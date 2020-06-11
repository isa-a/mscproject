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




def stochastic(time, X0, Br, Bc, Dr, Dc):
    R, C = X0
    return[((Br - Dr) * dt + (sqrt(Br + Dr)) * dWt), ((Bc - Dc) * dt + (sqrt(Bc + Dc)) * dWt)]


def stochasticR(Br, Dr):
    return((Br - Dr))
    
def stochasticR2(Br, Dr):
    return((sqrt(Br + Dr)))

Br = 1.0
Dr = 0.8
tspan = [0,100]
x0 = 0.1


result = itoint(stochasticR, x0, y0, tspan)




###############################################################################
def tauleap(Tinit, Tmax, R, C, K, rR, rC, alphaRC, alphaCR, tau = 0.001):
    ta = []
    Ra = []
    Ca = []

    t = 0
    R = R
    C = C
    while (t < Tmax - Tinit):
        # Previous step
        ta.append(t)
        Ra.append(R)
        Ca.append(C)

        Br = rR * R
        Bc = rC * C
        Dr = rR/K * R * (R + alphaRC * C)
        Dc = rC/K * C * (C + alphaCR * R)

    #    t += tau

      #  R += (np.random.poisson(B_R * tau, 1) - np.random.poisson(D_R * tau, 1))[0]
        #C += (np.random.poisson(B_C * tau, 1) - np.random.poisson(D_C * tau, 1))[0]

       # if R < 0:
         #   R = 0
      #  if C < 0:
        #    C = 0

   # ta = [t + t_init for t in ta]
  #  return(ta, Ra, Ca)



    
    
    
    
    
    
    
    
    
    