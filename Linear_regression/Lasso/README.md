# Lasso

We can use two way to find an optimal, one is gradient descent (as [Alogorithm 2](https://github.com/bochendong/Machine-learning/blob/master/Linear_regression/Lasso/README.md#algorithm-2)) and another is [Alogorithm 1](https://github.com/bochendong/Machine-learning/blob/master/Linear_regression/Lasso/README.md#algorithm-1)
### The Loss function of Lasso with L1 Regularization is:

![Lasso1](https://github.com/bochendong/Machine-learning/raw/master/Linear_regression/image/Lasso1.png)

### The Algotrithm of Lasso is :

![Lasso2](https://github.com/bochendong/Machine-learning/raw/master/Linear_regression/image/Lasso2.png)

## Algorithm 1:
### Giving a sample fact that:

![Lasso3](https://github.com/bochendong/Machine-learning/raw/master/Linear_regression/image/Lasso3.png)

### We can perform line 3 in above algorithm to O(n) time and space using equation(6) in following ways:

![Lasso4](https://github.com/bochendong/Machine-learning/raw/master/Linear_regression/image/Lasso4.png)

### Then the output of this algorithm is:
Note that we did not use gradient desencent, so the loss function is not monotonically decreasing
![Lasso4](https://github.com/bochendong/Machine-learning/raw/master/Linear_regression/image/figure_1.png)

## Algorithm 2:
### The Loss function of Lasso with L2 Regularization is:
![Lasso5](https://github.com/bochendong/Machine-learning/raw/master/Linear_regression/image/Lasso5.png)

Using Gradient desencent, take the derivative of the loss function w.r.t w and b, set max round be 100,000, lambda = 10 and learning rate = 0.000001

### Then the output of this algorithm is:
![Lasso4](https://github.com/bochendong/Machine-learning/raw/master/Linear_regression/image/figure_2.png)
