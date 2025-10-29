"""
Practice implementation of K-Nearest Neighbors
TODO: Complete the implementation of KNN class

HINTS - Useful functions for this implementation:
- np.argsort(array): Returns indices that would sort an array
- np.bincount(array): Count number of occurrences of each value
- euclidean_distance(x1, x2): Calculate Euclidean distance between two points
"""

import numpy as np
from mlfromscratch.utils import euclidean_distance


class KNN:
    """
    K Nearest Neighbors classifier.
    
    Parameters:
    -----------
    k: int
        The number of closest neighbors that will determine the class of the 
        sample that we wish to predict.
    """
    
    def __init__(self, k=5):
        """
        Initialize the KNN classifier.
        
        TODO: Initialize the following attributes:
        - self.k: number of neighbors to consider
        """
        # TODO: Implement initialization
        pass
    
    def _vote(self, neighbor_labels):
        """
        Return the most common class among the neighbor samples.
        
        Parameters:
        -----------
        neighbor_labels: array-like
            Labels of the k nearest neighbors
            
        Returns:
        --------
        most_common_label: int
            The most frequent label among neighbors
            
        TODO: Implement voting mechanism
        Hint: Use np.bincount to count occurrences of each label
        """
        # TODO: Implement voting
        pass
    
    def predict(self, X_test, X_train, y_train):
        """
        Make predictions for test samples.
        
        Parameters:
        -----------
        X_test: array-like, shape = [n_test_samples, n_features]
            Test samples
        X_train: array-like, shape = [n_train_samples, n_features]
            Training samples  
        y_train: array-like, shape = [n_train_samples]
            Training labels
            
        Returns:
        --------
        y_pred: array-like, shape = [n_test_samples]
            Predicted labels for test samples
            
        TODO: Implement prediction algorithm
        1. For each test sample:
           - Calculate distances to all training samples
           - Find k nearest neighbors
           - Use voting to determine predicted class
        """
        # TODO: Implement prediction
        pass