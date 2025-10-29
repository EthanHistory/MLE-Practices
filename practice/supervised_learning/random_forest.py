"""
Practice implementation of Random Forest
TODO: Complete the implementation of RandomForest class

HINTS - Useful functions for this implementation:
- np.random.choice(array, size, replace): Random sampling with/without replacement
- np.random.randint(low, high, size): Generate random integers
- np.mean(array, axis): Calculate mean along axis
- np.array([list of predictions]): Convert list to numpy array
- len(array): Get length of array
- range(n): Generate sequence of numbers
- DecisionTree/ClassificationTree: Import from your decision_tree module
- For classification: Use majority vote (most frequent prediction)
- For regression: Use average of predictions
"""

import numpy as np
# TODO: You may need to import DecisionTree classes


class RandomForest:
    """
    Random Forest classifier/regressor.
    
    Parameters:
    -----------
    n_estimators: int
        The number of classification trees that are used.
    max_features: int
        The maximum number of features that the classification trees are allowed to
        use.
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_gain: float
        The minimum impurity required to split the tree further.
    max_depth: int
        The maximum depth of a tree.
    """
    
    def __init__(self, n_estimators=100, max_features=None, min_samples_split=2,
                 min_gain=0, max_depth=float("inf")):
        """
        TODO: Initialize Random Forest parameters
        """
        # TODO: Implement initialization
        pass

    def fit(self, X, y):
        """
        Fit the Random Forest to training data.
        
        Parameters:
        -----------
        X: array-like, shape = [n_samples, n_features]
            Training data
        y: array-like, shape = [n_samples]
            Target values
            
        TODO: Implement training algorithm
        1. Create list of decision trees
        2. For each tree:
           - Sample data with replacement (bootstrap)
           - Sample features randomly
           - Train tree on sampled data and features
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
            Predicted class labels or values
            
        TODO: Make predictions using ensemble of trees
        1. Get predictions from all trees
        2. For classification: use majority vote
        3. For regression: use average
        """
        # TODO: Implement prediction
        pass