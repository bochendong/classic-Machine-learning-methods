import numpy as np
import collections
from collections import Counter

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
  """
  def compute_distances_two_loops(self, X):
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    print(X[0])
    print(X_train[0])
    for i in range(num_test):
      for j in range(num_train):
        dists[i][j] = np.sqrt(np.sum(np.square((X[i] - self.X_train[j]))))

    return dists

  def compute_distances_one_loop(self, X):
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))

    for i in range(num_test):
      point_diff = X[i] - self.X_train
      square = np.square(point_diff)
      summation = np.sum(square, axis=1)
      dists[i,:] = np.sqrt(summation)

    return dists

  ###
  # A = |a b| B = |e f|
  #     |c d|     |g h|
  # where (a, b), (c, d) are points in training set, (e, f), (g, h) are points in test set
  # We want to calculate the matrix of distances between test points and training points, such that:
  # Matrix of distance = np.sqrt(| [(a - e)^2 + (b - f)^2] [(a - g)^2 + (b - h)^2] |)
  #                              | [(c - e)^2 + (d - f)^2] [(c - g)^2 + (d - h)^2] |

  # Then, we can consider:
  # | [(a - e)^2 + (b - f)^2] [(a - g)^2 + (b - h)^2] |
  # | [(c - e)^2 + (d - f)^2] [(c - g)^2 + (d - h)^2] |
  # =
  # | [a^2 + e^2 + b^2 + f^2] [a^2 + g^2 + b^2 + h^2] | - 2| ae + bf  ag + bh |
  # | [c^2 + e^2 + d^2 + f^2] [c^2 + g^2 + d^2 + h^2] |    | ce + df  cg + dh |
  # = 
  # | [a^2 + e^2 + b^2 + f^2] [a^2 + g^2 + b^2 + h^2] | - 2AB^T
  # | [c^2 + e^2 + d^2 + f^2] [c^2 + g^2 + d^2 + h^2] |    
  
  # Hence
  # A = |a b| B = |e f|
  #     |c d|     |g h|
  #    
  # A*A = |a^2 b^2| B*B = |e^2 f^2|
  #        |c^2 d^2|       |g^2 h^2|
  #
  # C = np.sum(A*A, axis = 1) = |a^2 + b^2|
  #                             |c^2 + d^2|
  #
  # D = np.sum(B*B, axis = 1) = |e^2 + f^2|
  #                             |g^2 + h^2|

  # E = np.reshap(D, [2, 1]) = |e^2 + f^2 g^2 + h^2|

  # E + C = | [a^2 + b^2 + e^2 + f^2] [a^2 + b^2 + g^2 + h^2] |
  #         | [c^2 + d^2 + e^2 + f^2] [c^2 + d^2 + g^2 + h^2] |
    
  # E + C - 2 AB = | [a^2 + b^2 + e^2 + f^2 - 2ae - 2bf] [a^2 + b^2 + g^2 + h^2 - 2ag - 2bh] |
  #                | [c^2 + d^2 + e^2 + f^2 - 2ce - 2df] [c^2 + d^2 + g^2 + h^2 - 2cg - 2dh] |
  #              = | [(a - e)^2 + (b - f)^2] [(a - g)^2 + (b - h)^2] |
  #              = | [(c - e)^2 + (d - f)^2] [(c - g)^2 + (d - h)^2] |

  def compute_distances_no_loops(self, X):
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 

    dists = np.reshape(np.sum(X**2, axis=1), [num_test,1]) + np.sum(self.X_train**2, axis=1) \
            - 2 * np.matmul(X, self.X_train.T)
    dists = np.sqrt(dists)

    return dists



  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      closest_y = self.y_train[np.argsort(-dists[i])][:k]
      print(closest_y)
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################

      y_pred[i] = np.bincount(closest_y).argmax()
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred





