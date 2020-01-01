from numpy import *
import matplotlib.pyplot as plt
import math

class LR:
    w = -4
    b = -120
    y_pred = []
    loss = []
    
    def __init__(self, x, y, learning_rate, iteration):
        self.x = x
        self.y = y
        self.eta = learning_rate
        self.iter = iteration
    
    def loss_cal (self):
        loss_sum = 0;
        for i in range (len (self.x)):
            y_temp = self.x[i] * LR.w + LR.b
            loss_sum += abs(self.y[i] - y_temp)
        return loss_sum
    
    def learn (self):
        for i in range (0, self.iter):
            b_grad = 0.0
            w_grad = 0.0

            for n in range (len (x_data)):
                b_grad = b_grad - 2.0 * (self.y[n] - LR.b - LR.w * self.x[n])
                w_grad = w_grad - 2.0 * (self.y[n] - LR.b - LR.w * self.x[n]) * self.x[n]
            
            LR.b = LR.b - self.eta * b_grad
            LR.w = LR.w - self.eta * w_grad
        
            LR.loss.append(LR.loss_cal())
        
            for i in range (len (self.x)):
                LR.y_pred.append(LR.w * self.x[i] + Lin_Reg.b)
        
    def show_points(self):
        plt.plot(self.x, self.y, 'o')
        plt.xlim(0, 700)
        plt.ylim(0, 1700)
        plt.plot(self.x, LR.y_pred)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('linear_regression')
        plt.show()
        
        
    def show_loss(self):
        loss_x = range (0,100000)
        plt.title('LOSS')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.plot(loss_x, LR.loss,'o')
        plt.show()
            
x_data = [338., 333., 328., 207., 226., 25., 179., 60., 208., 606.]
y_data = [640., 633., 619., 393., 428., 27., 193., 66., 226., 1591.]

A = LR (x_data, y_data, 0.000001, 100000)
A.show_loss()
