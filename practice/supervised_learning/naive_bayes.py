"""
Practice implementation of Naive Bayes
TODO: Complete the implementation of NaiveBayes class

HINTS - Useful functions for this implementation:
- np.unique(array): Find unique elements in an array
- np.where(condition): Return indices where condition is True
- array.T: Transpose of array (for iterating over columns)
- col.mean(): Calculate mean of array
- col.var(): Calculate variance of array
- math.sqrt(x): Square root function
- math.pi: Pi constant
- math.exp(x): Exponential function
- math.pow(x, y): x raised to power y
- np.mean(condition): Calculate proportion of True values
- np.argmax(array): Index of maximum value
- zip(iter1, iter2): Iterate over multiple iterables simultaneously
"""

import numpy as np
import math


class NaiveBayes:
    """
    The Gaussian Naive Bayes classifier.
    
    TODO: Implement Gaussian Naive Bayes algorithm
    """
    
    def fit(self, X, y):
        """
        Fit the Naive Bayes classifier.
        
        Parameters:
        -----------
        X: array-like, shape = [n_samples, n_features]
            Training data
        y: array-like, shape = [n_samples]
            Target values
            
        TODO: Calculate mean and variance for each feature for each class
        1. Store X and y
        2. Find unique classes
        3. For each class, calculate mean and variance of each feature
        """
        # TODO: Implement fitting
        pass

    def _calculate_likelihood(self, mean, var, x):
        """
        Gaussian likelihood of the data x given mean and var.
        
        Parameters:
        -----------
        mean: float
            Mean of the feature for the class
        var: float
            Variance of the feature for the class
        x: float
            Feature value
            
        Returns:
        --------
        likelihood: float
            Gaussian likelihood
            
        TODO: Calculate Gaussian probability density
        P(x) = (1/sqrt(2*pi*var)) * exp(-(x-mean)^2 / (2*var))
        Add small epsilon to prevent division by zero
        """
        # TODO: Implement likelihood calculation
        pass

    def _calculate_prior(self, c):
        """
        Calculate the prior of class c.
        
        Parameters:
        -----------
        c: class label
            The class to calculate prior for
            
        Returns:
        --------
        prior: float
            Prior probability of class c
            
        TODO: Calculate P(class = c) = count(class = c) / total_samples
        """
        # TODO: Implement prior calculation
        pass

    def _classify(self, sample):
        """
        Classification using Bayes Rule P(Y|X) = P(X|Y)*P(Y)/P(X).
        
        Parameters:
        -----------
        sample: array-like
            Single sample to classify
            
        Returns:
        --------
        predicted_class: class label
            Predicted class for the sample
            
        TODO: Implement classification
        1. For each class, calculate posterior probability
        2. Posterior = Prior * Product of likelihoods for all features
        3. Return class with highest posterior
        """
        # TODO: Implement classification
        pass

    def predict(self, X):
        """
        Predict the class labels of the samples in X.
        
        Parameters:
        -----------
        X: array-like, shape = [n_samples, n_features]
            Samples to predict
            
        Returns:
        --------
        y_pred: array-like, shape = [n_samples]
            Predicted class labels
            
        TODO: Apply _classify to each sample
        """
        # TODO: Implement prediction
        pass