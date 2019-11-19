#! /usr/bin/env python3
from numpy import *
import numpy as np
import csv
import matplotlib.pyplot as plt
import time


# generate data
x = np.random.rand(2, 100)

y = []
i = 0

for x_axis in x[0]:
	if (x_axis/ 2 < x[1][i]):
		y.append(-1)
	else:
		y.append(1)
	i += 1

y = np.array(y).reshape(100,1)



# do prediction
def prediction(w, x, y, n):
	correct = 0.0

	for i in range (0, n):
		if (np.dot(w.T, x[:,i]) >= 0 and y[i] > 0):
			correct += 1.0
		elif (np.dot(w.T, x[:,i]) <= 0 and y[i] < 0):
			correct += 1.0

	return correct / n

loss_history = []



# FROM HERE IS MAIN FUNCTION
def sigmoid(w, x_i):
    return 1.0 / (1.0 + np.exp(- np.dot(w.T, x_i)))


def logistic_regression(x , y):
	n = len(y)
	d = len(x)

	w = np.zeros(d)

	eta = 0.001

	for k in range (0, 10000):
		g = np.zeros((d,1))

		for i in range (0, n):
			p_i = sigmoid(w, x[:,i])
			g += ((p_i - ((1.0 + y[i]) / 2.0)) * x[:,i]).reshape(d, 1)


		w = (w - (eta * g).T)[0]
		this_round_loss = prediction(w, x, y, n)
		loss_history.append(this_round_loss)
		print (w)
		if (np.linalg.norm(eta * g, 1) <= 0.001):
			break;
	return w

w = logistic_regression (x, y)
print (w)



# plot data 
x_l = range(0,len(y))
y_l = []

for i in x_l:
	y_i = 0 - w[0]*i / w[1]
	y_l.append(y_i)

plt.plot(x_l,y_l)

i = 0
for x_dim in x[0]:
	if (y[i] == -1):
		plt.plot(x_dim, x[1][i], 'o', color='g')
	else:
		plt.plot(x_dim, x[1][i], 'o',  color='red')
	i += 1
plt.ylim(0,1)
plt.xlim(0,1)
plt.show()

'''
length = len ( loss_history)
loss_x = range (0,length)
plt.plot(loss_x,loss_history, 'o')
plt.ylim(0,1)
plt.show()
'''




'''

A SAMPLE TEST
x = np.array([
	[2,    3, 4,   2,   10,   3,   2,   1,    7,   9 ,   21,  3,  14],
	[0.2 , 8, 1.5, 1.5, 3.2, 0.2, 2.2, 0.03, 3.2, 2.1, 13.5, 2.3, 3.2]
			])

y = np.array(
	[
	[-1],
	[1],
	[-1],
	[1],
	[-1],
	[-1],
	[1],
	[-1],
	[-1],
	[-1],
	[1],
	[1],
	[-1]
	]
)
'''
