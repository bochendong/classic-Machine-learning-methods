from numpy import *
import numpy as np
import csv
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import normalize
import math
from mpl_toolkits import mplot3d

# generate data
x = np.array([
	[2, 0.2],
	[3, 8],
	[4, 1.5],
	[2, 1.5],
	[10, 3.2],
	[3, 0.2],
	[2, 2.2],
	[1, 0.01],
	[7, 3.2],
	[9, 2.1],
	[21, 13.5],
	[3, 2.3],
	[14, 3.2],
			])

Y = np.array(
	[ [-1], [1], [-1], [1], [-1], [-1], [1], [-1], [-1], [-1], [1], [1], [-1] ]
)



def linear_kernel(X, z):
	return np.dot(X.T, z)

def KernelMatrix(X):
	K = np.matrix([[0]*len(X)]*len(X))

	for i in range(0, len(X)):
		for j in range(0, len(X)):
			K[i,j] = linear_kernel(X[i], X[j])

	return normalize(K)


def predict(X, x_test, alpha):
	pred = 0
	# pred = sum(alpha_i * k(x_i, x_test))
	i = 0
	for xi  in X:
		pred += alpha[i] * linear_kernel(xi, x_test)
		i += 1

	return pred

preds = []
def Test_Accuracy(X, y, alpha):
	correct = 0

	for i, yi in enumerate(y):
		pred = predict(X, X[i], alpha)
		print(pred)
		preds.append(pred)

		if pred * yi >= 0:
			correct += 1.0

	print(correct/len(y))

def sigmoid(t):
    return 1.0 / (1.0 + np.exp(- t))

def train_alpha (X, y, K, Lambda):
	eta = 0.001
	tol = 0.01

	n = len(X)
	alpha = np.zeros((n,1))

	for j in range (0, 8000):
		g = np.zeros((n,1))	

		for i in range (0, n):
			regularization = np.zeros((n,1))	
			for k in range (0, n):
				regularization += (2 * Lambda * alpha[i] * alpha[k]*  K[i]).reshape(n,1)
			g += (((-y[i] + sigmoid(y[i] * np.dot(alpha.T, K[:,i])) * y[i]) * K[:,i]).reshape(n,1)  + regularization)

		alpha = np.subtract(alpha, np.multiply(eta, g))
		# Test_Accuracy(X, y, alpha)
	return alpha


KM = KernelMatrix(x)
alp = train_alpha(x, Y, KM, 0.01)

print (alp)
Test_Accuracy(x, Y, alp)
print(preds)

w = np.array([0,0])

for i in range (0, len(x)):
	w = w + alp[i] * x[i]

print (w)

plt.axes(projection='3d')


