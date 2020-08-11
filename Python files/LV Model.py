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
def lotkavolterra(t, X, N, K, r, alpha):
                  #, Kr, Kc, Kc2, rR, rC, rC2, alphaCR, alphaRC, alphaC2R, alphaC2C, alphaCC2, alphaRC2):
    dXdt = []
    for i in range(N):
        sum = 0
        for j in range(N):
            sum = sum + alpha[j,i] * X[j]
        dX = r[i] * X[i] * ((K[i] - sum) / K[i])
        dXdt.append(dX)
    return(dXdt)
     

#create params
N = 2
t = [0,100]
X =  [100, 500] 
K = [1000,1000]
r = [1.5,1.5]
alpha = array([[1, 1.5],[1.3, 1]])
#alpha = array([[1, 0.5, 0.4], [0.4, 1, 0.63], [1.2, 0, 1]])

solve = solve_ivp(lotkavolterra, t, X, args=( N, K, r, alpha), dense_output=True)



def plotODE():
    t = linspace(0, 100, 100) #start and stop times and the no. of samples to generate
    Xsolve = solve.sol(t)
    plt.plot(t, Xsolve.T)
    plt.legend(['R', 'C','C2'])
    plt.xlabel('Time')
    plt.ylabel('Population size')
    plt.title('Lotka-Volterra System')
    plt.show()
plotODE()














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








#grwth(['rR', 'rC1', 'rC2', 'rC3'],
 #alpha['alpha0', 'alpha1', 'alpha2', 'alpha3'],
# X_list['R', 'C1', 'C2', 'C3'],
 #Capacity['Kr', 'Kc1', 'Kc2', 'Kc3'])



#Use lists
#Use vectors from numpy
#Use indexes
#Use dictionaries 




