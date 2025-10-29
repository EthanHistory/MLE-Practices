"""
Practice implementation of Regression models
TODO: Complete the implementation of regression classes

HINTS - Useful functions for this implementation:
- np.linalg.norm(array, ord=1): L1 norm (ord=1) or L2 norm (ord=2, default)
- np.sign(array): Element-wise sign function
- np.insert(array, obj, values, axis): Insert values along the given axis
- np.mean(array): Calculate mean of array elements
- np.random.uniform(low, high, size): Generate random numbers
- array.T: Transpose of array
- array.dot(other): Matrix multiplication
- np.linalg.svd(matrix): Singular value decomposition
- np.linalg.pinv(matrix): Moore-Penrose pseudoinverse
- normalize(X): Normalize features from mlfromscratch.utils
- polynomial_features(X, degree): Create polynomial features from mlfromscratch.utils
"""

import numpy as np
import math
from mlfromscratch.utils import normalize, polynomial_features


class l1_regularization:
    """
    Regularization for Lasso Regression
    
    TODO: Implement L1 regularization (absolute value penalty)
    """
    
    def __init__(self, alpha):
        """
        Parameters:
        -----------
        alpha: float
            Regularization strength
        """
        # TODO: Implement initialization
        pass
    
    def __call__(self, w):
        """
        Calculate L1 penalty.
        
        Parameters:
        -----------
        w: array-like
            Model weights
            
        Returns:
        --------
        penalty: float
            L1 regularization penalty
            
        TODO: Return alpha * L1_norm(w)
        Hint: Use np.linalg.norm with appropriate ord parameter
        """
        # TODO: Implement L1 penalty calculation
        pass

    def grad(self, w):
        """
        Calculate gradient of L1 penalty.
        
        Parameters:
        -----------
        w: array-like
            Model weights
            
        Returns:
        --------
        gradient: array-like
            Gradient of L1 penalty w.r.t. weights
            
        TODO: Return alpha * sign(w)
        """
        # TODO: Implement L1 gradient
        pass


class l2_regularization:
    """
    Regularization for Ridge Regression
    
    TODO: Implement L2 regularization (squared penalty)
    """
    
    def __init__(self, alpha):
        # TODO: Implement initialization
        pass
    
    def __call__(self, w):
        """
        Calculate L2 penalty.
        
        TODO: Return alpha * 0.5 * w^T * w
        """
        # TODO: Implement L2 penalty calculation
        pass

    def grad(self, w):
        """
        Calculate gradient of L2 penalty.
        
        TODO: Return alpha * w
        """
        # TODO: Implement L2 gradient
        pass


class Regression:
    """
    Base regression model.
    
    Parameters:
    -----------
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    
    def __init__(self, n_iterations, learning_rate):
        # TODO: Implement initialization
        pass

    def initialize_weights(self, n_features):
        """
        Initialize weights randomly [-1/N, 1/N]
        
        Parameters:
        -----------
        n_features: int
            Number of features
            
        TODO: Initialize self.w with random values in range [-1/sqrt(n_features), 1/sqrt(n_features)]
        """
        # TODO: Implement weight initialization
        pass

    def fit(self, X, y):
        """
        Fit the regression model.
        
        Parameters:
        -----------
        X: array-like, shape = [n_samples, n_features]
            Training data
        y: array-like, shape = [n_samples]
            Target values
            
        TODO: Implement training algorithm
        1. Insert bias column (ones) to X
        2. Initialize weights
        3. For each iteration:
           - Calculate predictions
           - Calculate MSE loss with regularization
           - Calculate gradients
           - Update weights
        """
        # TODO: Implement training
        pass

    def predict(self, X):
        """
        Make predictions.
        
        Parameters:
        -----------
        X: array-like, shape = [n_samples, n_features]
            Test data
            
        Returns:
        --------
        y_pred: array-like, shape = [n_samples]
            Predicted values
            
        TODO: Implement prediction
        1. Insert bias column to X
        2. Return X.dot(self.w)
        """
        # TODO: Implement prediction
        pass


class LinearRegression(Regression):
    """
    Linear model.
    
    Parameters:
    -----------
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    gradient_descent: boolean
        True or false depending if gradient descent should be used when training.
    """
    
    def __init__(self, n_iterations=100, learning_rate=0.001, gradient_descent=True):
        """
        TODO: Initialize LinearRegression
        - Set gradient_descent parameter
        - Set regularization to no regularization (lambda functions returning 0)
        - Call parent constructor
        """
        # TODO: Implement initialization
        pass
    
    def fit(self, X, y):
        """
        Fit the model using either gradient descent or least squares.
        
        TODO: If not gradient_descent:
        - Use least squares solution: w = (X^T X)^(-1) X^T y
        - Remember to add bias column to X
        Else:
        - Use parent's fit method (gradient descent)
        """
        # TODO: Implement fitting
        pass


class LassoRegression(Regression):
    """
    Linear regression with L1 regularization.
    
    TODO: Implement Lasso regression with polynomial features and normalization
    """
    
    def __init__(self, degree, reg_factor, n_iterations=3000, learning_rate=0.01):
        # TODO: Implement initialization
        # Set self.degree and self.regularization (L1)
        pass

    def fit(self, X, y):
        """
        TODO: 
        1. Transform X using polynomial features
        2. Normalize the features
        3. Call parent's fit method
        """
        # TODO: Implement fitting
        pass

    def predict(self, X):
        """
        TODO:
        1. Transform X using polynomial features  
        2. Normalize the features
        3. Call parent's predict method
        """
        # TODO: Implement prediction
        pass


class RidgeRegression(Regression):
    """
    Linear regression with L2 regularization.
    
    TODO: Implement Ridge regression
    """
    
    def __init__(self, reg_factor, n_iterations=1000, learning_rate=0.001):
        # TODO: Implement initialization
        # Set self.regularization (L2)
        pass


class PolynomialRegression(Regression):
    """
    Polynomial regression.
    
    TODO: Implement polynomial regression (no regularization)
    """
    
    def __init__(self, degree, n_iterations=3000, learning_rate=0.001):
        # TODO: Implement initialization
        pass

    def fit(self, X, y):
        """
        TODO: Transform X to polynomial features before fitting
        """
        # TODO: Implement fitting
        pass

    def predict(self, X):
        """
        TODO: Transform X to polynomial features before prediction
        """
        # TODO: Implement prediction
        pass


class ElasticNet(Regression):
    """
    Linear regression with both L1 and L2 regularization.
    
    TODO: Implement ElasticNet regression
    """
    
    def __init__(self, degree=1, reg_factor=0.05, l1_ratio=0.5, n_iterations=3000, learning_rate=0.01):
        # TODO: Implement initialization
        # Use l1_l2_regularization
        pass

    def fit(self, X, y):
        """
        TODO: Apply polynomial transformation and normalization if needed
        """
        # TODO: Implement fitting
        pass

    def predict(self, X):
        """
        TODO: Apply same transformations as in fit
        """
        # TODO: Implement prediction
        pass