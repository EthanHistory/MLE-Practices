"""
Practice implementation of K-Nearest Neighbors
TODO: Complete the implementation of KNN class

HINTS - Useful functions for this implementation:
- np.argsort(array) -> array: Returns indices that would sort an array
- np.bincount(array) -> array: Count number of occurrences of each value
- euclidean_distance(x1, x2) -> float: Calculate Euclidean distance between two points
"""
from typing import List
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
        self.k = k
        pass
    
    def _vote(self, neighbor_labels: np.ndarray):
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
        counts = np.bincount(neighbor_labels)
        most_common_class = np.argmax(counts)
        return most_common_class
    
    def predict(self, X_test: np.ndarray, X_train: np.ndarray, y_train: np.ndarray):
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
        
        # Calculate distances to all training samples
        distances = np.empty(shape=[X_test.shape[0], X_train.shape[0]]) # [n_test_samples, n_train_samples]
        for i in range(distances.shape[0]):
            for j in range(distances.shape[1]):
                distances[i, j] = euclidean_distance(X_test[i, :], X_train[j, :])
        
        # Find k nearest neighbors
        k_nearest_neighbors = np.empty(shape=[X_test.shape[0], self.k], dtype=np.int64) # [n_test_samples, k]
        for i in range(k_nearest_neighbors.shape[0]):
            k_nearest_neighbors[i] = y_train[np.argsort(distances[i])[:self.k]]

        # Use voting to determine predicted class
        y_pred = np.empty(shape=[X_test.shape[0]]) # [n_test_samples]
        for i in range(X_test.shape[0]):
            y_pred[i] = self._vote(k_nearest_neighbors[i])

        return y_pred
    
# Feedback
# Original version is slightly better in robustness (forces integer labels in _vote) and uses less memory (no full distance matrix). Practice version adds type hints and clearer structure but wastes memory and misses label casting.