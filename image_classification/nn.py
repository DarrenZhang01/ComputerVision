# reference from stanford CS231n "http://cs231n.github.io/classification/#intro"

import numpy as np

class NearestNeighbor(object):

    def __init__(self):
        pass

    def train(self, X, y):
        """
        X is of dimension N x D where each row represents an image
        y is N dimensional where it represents N labels accordingly for each
        training example
        """
        # The nearest neighbor training classifier simply remembers all the training data
        self.Xtrain = X
        self.ytrain = y

    def predict(self, X):
        """
        X is N x D dimensional where each row is an example where we would like
        to predict labels for
        """
        # initialize the prediction data array
        ypred = np.zeros(X.shape[0], dtype = self.ytrain.dtype)
        # loop through each test example
        for i in range(X.shape[0]):
            # Find the nearest training image to the test image
            # using the L1 distance (sum of absolute differences)
            l1_distance_array = np.sum(np.abs(self.Xtrain - X[i, :]), axis = 1)
            # get the index with the smallest difference
            min_index = np.argmin(l1_distance_array)
            # put the prediction label into the prediction array
            ypred[i] = self.ytrain[min_index]

        return ypred
