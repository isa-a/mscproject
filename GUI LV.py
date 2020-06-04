#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 09:25:25 2020

@author: isa
"""

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numpy import *
import tkinter as tk
from tkinter import IntVar

#creating function with the LV equations within it, first listing all parameters

def startwindow():
    startwindow1 = tk.Tk()
    startwindow1.geometry('300x300')
    startwindow1.title('Enter values')
    
    #need values for - min time, max time, two y0 values, carrying capacity, growth rates for R and C, alphaCR and alphaR, no. of samples
    
    tk.Label(startwindow1, text = 'Start time', font = ('verdana')).grid(row=0)
    tk.Label(startwindow1, text = 'End time', font = ('verdana')).grid(row=1)
    tk.Label(startwindow1, text = 'R', font = ('verdana')).grid(row=2)
    tk.Label(startwindow1, text = 'C', font = ('verdana')).grid(row=3)
    tk.Label(startwindow1, text = 'Carrying capacity', font = ('verdana')).grid(row=4)
    tk.Label(startwindow1, text = 'R growth rate', font = ('verdana')).grid(row=5)
    tk.Label(startwindow1, text = 'C growth rate', font = ('verdana')).grid(row=6)
    tk.Label(startwindow1, text = 'alphaCR', font = ('verdana')).grid(row=7)
    tk.Label(startwindow1, text = 'alphaRC', font = ('verdana')).grid(row=8)
    tk.Label(startwindow1, text = 'No. of samples', font = ('verdana')).grid(row=9)
    
    getmintime = IntVar()
    mintimeBox = tk.Entry(startwindow1, textvariable = getmintime, width=3)
    mintimeBox.grid(row=0, column=1)           

    getmaxtime = IntVar()
    maxtimeBox = tk.Entry(startwindow1, textvariable = getmaxtime, width=3)
    maxtimeBox.grid(row=1, column=1)           

    getR = IntVar()
    rBox = tk.Entry(startwindow1, textvariable = getR, width=3)
    rBox.grid(row=2, column=1)           
 
    getC = IntVar()
    cBox = tk.Entry(startwindow1, textvariable = getC, width=3)
    cBox.grid(row=3, column=1)           
    
    getCap = IntVar()
    capBox = tk.Entry(startwindow1, textvariable = getCap, width=3)
    capBox.grid(row=4, column=1)           
  
    getRgrow = IntVar()
    rGrowBox = tk.Entry(startwindow1, textvariable = getRgrow, width=3)
    rGrowBox.grid(row=5, column=1)           
  
    getCgrow = IntVar()
    cGrowBox = tk.Entry(startwindow1, textvariable = getCgrow, width=3)
    cGrowBox.grid(row=6, column=1)           
    
    getalphaCR = IntVar()
    alphaCRBox = tk.Entry(startwindow1, textvariable = getalphaCR, width=3)
    alphaCRBox.grid(row=7, column=1)           
    
    getalphaRC = IntVar()
    alphaRCBox = tk.Entry(startwindow1, textvariable = getalphaRC, width=3)
    alphaRCBox.grid(row=8, column=1)           
  
    getSamples = IntVar()
    samplesBox = tk.Entry(startwindow1, textvariable = getSamples, width=3)
    samplesBox.grid(row=9, column=1)           
  
    
    readmin = getmintime.get()
    readmax = getmaxtime.get()
    readR = getR.get()
    readC = getC.get()
    readcap = getCap.get()
    readRgrow = getRgrow.get()
    readCgrow = getCgrow.get()
    readalphaCR = getalphaCR.get()
    readalphaRC = getalphaRC.get()
    readsamples = getSamples.get()
    
    intmin = int(readmin)
    intmax = int(readmax)
    intR = int(readR)
    intC = int(readC)
    intcap = int(readcap)
    intRgrow = int(readRgrow)
    intCgrow = int(readCgrow)
    intalphaCR = int(readalphaCR)
    intalphaRC = int(readalphaRC)
    intsamples = int(readsamples)


         
    def solveandplot():
        
        #X = readR, readC
        #lotkavolterra = [(readRgrow*readR)*((readcap - readR - (readalphaCR*readC)) / readcap), (readCgrow*readC)*((readcap - readC - (readalphaRC*readR)) / readcap)]

        def lotkavolterra(t, X, readcap, readRgrow, readCgrow, readalphaCR, readalphaRC):
            readR, readC = X
            return [(readRgrow*readR)*((readcap - readR - (readalphaCR*readC)) / readcap), (readCgrow*readC)*((readcap - readC - (readalphaRC*readR)) / readcap)]




        solve = solve_ivp(lotkavolterra, [readmin, readmax], [readR, readC], args=(readcap, readRgrow, readCgrow, readalphaRC, readalphaCR), dense_output=True) #passing through the values into the scipy solver
        
        t = linspace(readmin, readmax, readsamples) #start and stop times and the no. of samples to generate
        X = solve.sol(t)
        plt.plot(t, X.T)
        plt.legend(['R', 'C'])
        plt.title('Lotka-Volterra System')
        plt.savefig('Lotka-Volterra System.png')
        plt.show()



    btn = tk.Button(startwindow1, text='Plot graph', command = lambda: solveandplot(), font = ('verdana'))
    btn.grid(row=10)

    startwindow1.mainloop()
        
startwindow()
    


