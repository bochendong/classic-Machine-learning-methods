from numpy import *
import matplotlib.pyplot as plt
import math

x_data = [338., 333., 328., 207., 226., 25., 179., 60., 208., 606.]
y_data = [640., 633., 619., 393., 428., 27., 193., 66., 226., 1591.]

def loss_cal (x, y, weight, bias):
    loss_sum = 0;
    for i in range (len (x)):
        y_temp = x[i] * weight + bias
        loss_sum += abs(y[i] - y_temp)
    return loss_sum

def linear_regression (x_data, y_data, learning_rate, iteration):
    b = -120
    w = -4
    loss = []

    for i in range (0, 100000):

        b_grad = 0.0
        w_grad = 0.0

        # here use the gradient descent method:
        # \parial L   
        # --------- =  -2 * sum_{i = 1}^n (y_i - (b + w * x_i))
        # \parial b  

        # \parial L   
        # --------- =  -2 * sum_{i = 1}^n (y_i - (b + w * x_i)) * (x_i)
        # \parial w   

        for n in range (len (x_data)):
            b_grad = b_grad - 2.0 * (y_data[n] - b - w * x_data[n])
            w_grad = w_grad - 2.0 * (y_data[n] - b - w * x_data[n]) * x_data[n]
        
        # b_new = b - \eta * b_grad
        # b_new = w - \eta * w_grad
        b = b - learning_rate * b_grad
        w = w - learning_rate * w_grad

        loss.append(loss_cal (x_data, y_data, w, b))
        

    # plot the predict line and the orginal data:
         # this calculate the predict y
        y_pred = []
        for i in range (len (x_data)):
            y_pred.append(w * x_data[i] + b)

    plt.plot(x_data,y_data,'o')
    plt.xlim(0, 700)
    plt.ylim(0, 1700)
    plt.plot(x_data,y_pred)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('linear_regression')
    plt.show()

    # plot the chage of loss function
    loss_x = range (0,100000)
    plt.title('LOSS')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.plot(loss_x, loss,'o')
    plt.show()

linear_regression (x_data, y_data, 0.000001, 100000)
