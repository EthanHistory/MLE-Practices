"""
Practice implementation of Logistic Regression
TODO: Complete the implementation of LogisticRegression class

HINTS - Useful functions for this implementation:
- np.shape(array): Get the shape of an array
- math.sqrt(x): Square root function
- np.random.uniform(low, high, size): Generate random numbers from uniform distribution
- np.round(array): Round array elements to nearest integer
- array.astype(type): Convert array to specified type
- X.dot(weights): Matrix multiplication
- Sigmoid(): Activation function from mlfromscratch.deep_learning.activation_functions
- make_diagonal(): Create diagonal matrix from mlfromscratch.utils
"""

import numpy as np
import math
from mlfromscratch.utils import make_diagonal
from mlfromscratch.deep_learning.activation_functions import Sigmoid


class LogisticRegression:
    """
    Logistic Regression classifier.
    
    Parameters:
    -----------
    learning_rate: float
        The step length that will be taken when following the negative gradient during training.
    gradient_descent: boolean
        True or false depending if gradient descent should be used when training.
    """
    
    def __init__(self, learning_rate=0.1, gradient_descent=True):
        """
        Initialize the LogisticRegression model.
        
        TODO: Initialize the following attributes:
        - self.param: parameters/weights (will be set during fit)
        - self.learning_rate: learning rate for gradient descent
        - self.gradient_descent: whether to use gradient descent
        - self.sigmoid: sigmoid activation function (you may need to import this)
        """
        # TODO: Implement initialization
        pass
    
    def _initialize_parameters(self, X):
        """
        Initialize parameters randomly.
        
        Parameters:
        -----------
        X: array-like, shape = [n_samples, n_features]
            Training data
            
        TODO: Initialize self.param with random values between [-1/sqrt(n_features), 1/sqrt(n_features)]
        """
        # TODO: Implement parameter initialization
        pass
    
    def fit(self, X, y, n_iterations=4000):
        """
        Fit the logistic regression model to training data.
        
        Parameters:
        -----------
        X: array-like, shape = [n_samples, n_features]
            Training data
        y: array-like, shape = [n_samples]
            Target values
        n_iterations: int
            Number of training iterations
            
        TODO: Implement the training algorithm
        1. Initialize parameters
        2. For each iteration:
           - Make predictions using sigmoid function
           - Update parameters using gradient descent or batch optimization
        """
        # TODO: Implement training
        pass
    
    def predict(self, X):
        """
        Make predictions on test data.
        
        Parameters:
        -----------
        X: array-like, shape = [n_samples, n_features]
            Test data
            
        Returns:
        --------
        y_pred: array-like, shape = [n_samples]
            Predicted class labels (0 or 1)
            
        TODO: Implement prediction
        1. Calculate linear combination: X.dot(self.param)
        2. Apply sigmoid function
        3. Round to get binary predictions
        """
        # TODO: Implement prediction
        pass