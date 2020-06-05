#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 06:29:00 2020

@author: isa
"""
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numpy import *
import numpy as np

#creating function with the LV equations within it, first listing all parameters
def lotkavolterra(t, X, Kr, Kc, Kc2, rR, rC, rC2, alphaCR, alphaRC, alphaC2R, alphaC2C, alphaCC2, alphaRC2):
     R, C, C2 = X
     return [(rR*R)*((Kr - R - (alphaCR*C) - (alphaC2R*C2)) / Kr),     (rC*C)*((Kc - C - (alphaRC*R) - (alphaC2C*C2)) / Kc),     (rC2*C2)*((Kc2 - C2 - (alphaCC2*C) - (alphaRC2*R)) / Kc2)]



solve = solve_ivp(lotkavolterra, [0, 100], [50, 200, 350], args=(1000, 500, 430, 1, 1, 1, 0.8, 0.8, 0.8, 0.46, 0.52, 0.36), dense_output=True) #passing through the values into the scipy solver



t = linspace(0, 100, 100) #start and stop times and the no. of samples to generate
X = solve.sol(t)
plt.plot(t, X.T)
plt.legend(['R', 'C','C2'])
plt.title('Lotka-Volterra System')
plt.show()


#############################################################################################

Number = 4
#no. of comp terms in each equation is N-1

for i in range(1, Number):
    grwth =  [f"rC{i}" for i in range(1, Number)]
    alpha = [f"alpha{i}" for i in range(1, Number)]
    X_List = [f"C{i}" for i in range(1, Number)]
    Capacity = [f"Kc{i}" for i in range(1, Number)]
print(grwth)
print(alpha)
print(X_List)
print(Capacity)


params = (grwth,alpha,X_List, Capacity)

#eqn = (grwth[0]*X_List[0])*((Capacity[0] - X_List[0] - ))


sum = 0
for j in range (0, Number-1):
    sum = alpha{j}*C{j}







#grwth(['rR', 'rC1', 'rC2', 'rC3'],
 #alpha['alpha0', 'alpha1', 'alpha2', 'alpha3'],
# X_list['R', 'C1', 'C2', 'C3'],
 #Capacity['Kr', 'Kc1', 'Kc2', 'Kc3'])



#Use lists
#Use vectors from numpy
#Use indexes
#Use dictionaries 




