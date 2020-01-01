#! /usr/bin/env python3
from numpy import *
import numpy as np
import csv
import matplotlib.pyplot as plt

class Preception:
    max_pass = 500
    mistake = [0] * max_pass
    
    def __init__(self, x, y):
        n = len(y)
        d = len(x)

        # init w as [0 0 ... 0 0]
        w = np.array([0]* d).reshape(d,1)
        # init b = 0
        b = 0

        for t in range(0, Preception.max_pass):
            for i in range (0, n):
                if (y[i] * (np.vdot(x[:,i], w) + b) <= 0) :
                    w = w + (y[i] * x[:,i]).reshape(d,1)
                    b = b + y[i]
                    Preception.mistake[t] = Preception.mistake[t] + 1

    def show(self):
        x_axis = []
        for i in range(0,500):
            x_axis.append(i)

        plt.scatter(x_axis,Preception.mistake)
        plt.xlabel('Passes')
        plt.ylabel('Mistakes')
        plt.title('Perceptron')
        plt.show()
            
x = np.loadtxt(open("spambase_x.csv", "rb"), delimiter=",")
y = np.loadtxt(open("spambase_y.csv", "rb"), delimiter=",")

prec = Preception(x, y)
prec.show()
