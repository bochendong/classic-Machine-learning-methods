#! /usr/bin/env python3
from numpy import *
import numpy as np
import csv
import matplotlib.pyplot as plt

x = np.loadtxt(open("spambase_x.csv", "rb"), delimiter=",")
y = np.loadtxt(open("spambase_y.csv", "rb"), delimiter=",")

'''
Main idea:
for t = 1, 2, ..., max_pass do:
    for i = 1, 2, ..., n do:
        if (y_i<x_i, w> + b <= 0) then:
            w = w + y_i x_i
            b = b + y_i
            mistake[t] = mistake[t] + 1
'''

def preception (x, y):
    n = len(y)
    d = len(x)

    # init w as [0 0 ... 0 0]
    w = np.array([0]* d).reshape(d,1)
    # init b = 0
    b = 0

    max_pass = 500
    mistake = [0] * max_pass

    for t in range(0, max_pass):
        for i in range (0, n):
            if (y[i] * (np.vdot(x[:,i], w) + b) <= 0) :
                w = w + (y[i] * x[:,i]).reshape(d,1)
                b = b + y[i]
                mistake[t] = mistake[t] + 1
                
    x_axis = []
    for i in range(0,500):
        x_axis.append(i)
        
    plt.scatter(x_axis,mistake)
    plt.xlabel('Passes')
    plt.ylabel('Mistakes')
    plt.title('Perceptron')
    plt.show()                

preception (x, y)
