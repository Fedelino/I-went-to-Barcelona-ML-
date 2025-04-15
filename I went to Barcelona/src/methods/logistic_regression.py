import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.w = None

    def f_softmax(self, data):
        """
        Softmax function for multi-class logistic regression.
        
        Arguments:
            data (array): Input data of shape (N, D)
        Returns:
            array of shape (N, C): Probability array where each value is in the
                range [0, 1] and each row sums to 1.
                The row i corresponds to the prediction of the ith data sample, and 
                the column j to the jth class. So element [i, j] is P(y_i=k | x_i, W)
        """
        num = np.exp(data @ self.w)
        den = np.sum(num, axis = 1, keepdims = True)
        return num / den

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        D = training_data.shape[1]  # number of features
        C = training_labels.shape[1]  # number of classes

        # Computation of the weights
        self.w = np.random.normal(0, 0.1, (D, C))
        for it in range(self.max_iters):
            gradient = training_data.T @ (self.f_softmax(training_data) - training_labels)
            self.w = self.w - self.lr * gradient

        pred_labels = self.predict(training_data)
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        pred_labels = np.argmax(self.f_softmax(test_data), axis = 1, keepdims = True).flatten()
        return pred_labels
