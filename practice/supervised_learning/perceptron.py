"""
Practice implementation of Perceptron
TODO: Complete the implementation of Perceptron class

HINTS - Useful functions for this implementation:
- np.shape(array): Get the shape of an array
- math.sqrt(x): Square root function
- np.random.uniform(low, high, size): Generate random numbers
- np.zeros(shape): Create array filled with zeros
- X.dot(weights): Matrix multiplication
- np.sum(array, axis, keepdims): Sum along specified axis
- Sigmoid(): Activation function from mlfromscratch.deep_learning.activation_functions
- SquareLoss(): Loss function from mlfromscratch.deep_learning.loss_functions
- CrossEntropy(): Loss function from mlfromscratch.deep_learning.loss_functions
"""

import math
import numpy as np
from mlfromscratch.deep_learning.activation_functions import Sigmoid
from mlfromscratch.deep_learning.loss_functions import SquareLoss


class Perceptron:
    """
    The Perceptron. One layer neural network classifier.

    Parameters:
    -----------
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    activation_function: class
        The activation that shall be used for each neuron.
    loss: class
        The loss function used to assess the model's performance.
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    
    def __init__(self, n_iterations=20000, activation_function=None, loss=None, learning_rate=0.01):
        """
        TODO: Initialize perceptron parameters
        - self.n_iterations: number of training iterations
        - self.learning_rate: learning rate
        - self.loss: loss function instance
        - self.activation_func: activation function instance
        - self.W: weights (initialized during fit)
        - self.w0: bias (initialized during fit)
        """
        # TODO: Implement initialization
        pass

    def fit(self, X, y):
        """
        Fit the perceptron to training data.
        
        Parameters:
        -----------
        X: array-like, shape = [n_samples, n_features]
            Training data
        y: array-like, shape = [n_samples, n_outputs]
            Target values (should be one-hot encoded for multi-class)
            
        TODO: Implement training algorithm
        1. Initialize weights and bias randomly
        2. For each iteration:
           - Calculate linear output: X.dot(W) + w0
           - Apply activation function
           - Calculate loss gradient
           - Update weights using gradient descent
        """
        # TODO: Implement training
        pass

    def predict(self, X):
        """
        Use the trained model to predict labels of X.
        
        Parameters:
        -----------
        X: array-like, shape = [n_samples, n_features]
            Test data
            
        Returns:
        --------
        y_pred: array-like, shape = [n_samples, n_outputs]
            Predicted values
            
        TODO: Make predictions using trained weights
        1. Calculate linear output: X.dot(W) + w0  
        2. Apply activation function
        """
        # TODO: Implement prediction
        pass