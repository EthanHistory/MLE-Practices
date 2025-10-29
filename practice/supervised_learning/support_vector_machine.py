"""
Practice implementation of Support Vector Machine
TODO: Complete the implementation of SupportVectorMachine class

HINTS - Useful functions for this implementation:
- cvxopt.matrix(array): Convert numpy array to cvxopt matrix
- cvxopt.solvers.qp(P, q, G, h, A, b): Solve quadratic programming problem
- np.outer(a, b): Outer product of two vectors
- np.ones(shape): Create array filled with ones
- np.zeros(shape): Create array filled with zeros
- np.diag(array): Create diagonal matrix or extract diagonal
- kernel functions: linear, polynomial, rbf kernels from mlfromscratch.utils
- Lagrange multipliers: alpha values from QP solver
- Support vectors: training samples where alpha > threshold
"""

import numpy as np
import cvxopt
# TODO: You may need to import kernel functions from utils


class SupportVectorMachine:
    """
    The Support Vector Machine classifier.
    Uses cvxopt to solve the quadratic optimization problem.
    
    Parameters:
    -----------
    C: float
        Penalty parameter C of the error term.
    kernel: function
        Kernel function. Can be either polynomial, rbf or linear.
    power: int
        The degree of the polynomial kernel. Will be ignored by the other kernel functions.
    gamma: float
        Kernel coefficient for 'rbf' and 'polynomial' kernels. Will be ignored by linear kernel.
    coef: float
        Independent term in kernel function. It is only significant in 'polynomial' kernels.
    """
    
    def __init__(self, C=1, kernel=None, power=4, gamma=None, coef=4):
        """
        TODO: Initialize SVM parameters
        """
        # TODO: Implement initialization
        pass

    def fit(self, X, y):
        """
        Fit the SVM to training data.
        
        Parameters:
        -----------
        X: array-like, shape = [n_samples, n_features]
            Training data
        y: array-like, shape = [n_samples]
            Target values (should be -1 or 1)
            
        TODO: Implement SVM training using quadratic programming
        1. Setup quadratic programming problem
        2. Solve for Lagrange multipliers using cvxopt
        3. Calculate support vectors and bias
        """
        # TODO: Implement training
        pass

    def predict(self, X):
        """
        Predict class for X.
        
        Parameters:
        -----------
        X: array-like, shape = [n_samples, n_features]
            Test samples
            
        Returns:
        --------
        y_pred: array-like, shape = [n_samples]
            Predicted class labels
            
        TODO: Make predictions using support vectors
        1. Calculate kernel values between X and support vectors
        2. Compute decision function
        3. Return sign of decision function
        """
        # TODO: Implement prediction
        pass