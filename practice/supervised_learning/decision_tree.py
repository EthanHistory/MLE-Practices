"""
Practice implementation of Decision Tree
TODO: Complete the implementation of decision tree classes

HINTS - Useful functions for this implementation:
- np.shape(array): Get the shape of an array
- np.expand_dims(array, axis): Add a new axis to array
- np.concatenate((arr1, arr2), axis): Join arrays along existing axis
- np.unique(array): Find unique elements
- len(array): Get length of array
- float("inf"): Infinity value
- divide_on_feature(Xy, feature_i, threshold): Split data on feature from mlfromscratch.utils
- calculate_entropy(y): Calculate entropy from mlfromscratch.utils
- calculate_variance(y): Calculate variance from mlfromscratch.utils
- np.mean(array): Calculate mean
- np.bincount(array): Count occurrences of each value
"""

import numpy as np
from mlfromscratch.utils import divide_on_feature, calculate_entropy, calculate_variance


class DecisionNode:
    """
    Class that represents a decision node or leaf in the decision tree

    Parameters:
    -----------
    feature_i: int
        Feature index which we want to use as the threshold measure.
    threshold: float
        The value that we will compare feature values at feature_i against to
        determine the prediction.
    value: float
        The class prediction if classification tree, or float value if regression tree.
    true_branch: DecisionNode
        Next decision node for samples where features value met the threshold.
    false_branch: DecisionNode
        Next decision node for samples where features value did not meet the threshold.
    """
    
    def __init__(self, feature_i=None, threshold=None,
                 value=None, true_branch=None, false_branch=None):
        """
        TODO: Initialize decision node attributes
        """
        # TODO: Implement initialization
        pass


class DecisionTree:
    """
    Super class of RegressionTree and ClassificationTree.

    Parameters:
    -----------
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float
        The minimum impurity required to split the tree further.
    max_depth: int
        The maximum depth of a tree.
    loss: function
        Loss function that is used for Gradient Boosting models to calculate impurity.
    """
    
    def __init__(self, min_samples_split=2, min_impurity=1e-7,
                 max_depth=float("inf"), loss=None):
        """
        TODO: Initialize decision tree parameters
        - self.root: root node (initially None)
        - self.min_samples_split: minimum samples to split
        - self.min_impurity: minimum impurity to split
        - self.max_depth: maximum tree depth
        - self._impurity_calculation: function to calculate impurity (set in subclasses)
        - self._leaf_value_calculation: function to calculate leaf values (set in subclasses)
        - self.one_dim: whether y is one-dimensional
        - self.loss: loss function for gradient boosting
        """
        # TODO: Implement initialization
        pass

    def fit(self, X, y, loss=None):
        """
        Build decision tree.
        
        Parameters:
        -----------
        X: array-like, shape = [n_samples, n_features]
            Training data
        y: array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target values
        loss: function, optional
            Loss function for gradient boosting
            
        TODO: 
        1. Set self.one_dim based on y shape
        2. Build tree using _build_tree method
        """
        # TODO: Implement fitting
        pass

    def _build_tree(self, X, y, current_depth=0):
        """
        Recursive method which builds out the decision tree and splits X and respective y
        on the feature of X which (based on impurity) best separates the data.
        
        Parameters:
        -----------
        X: array-like
            Training features
        y: array-like
            Training targets
        current_depth: int
            Current depth in the tree
            
        Returns:
        --------
        DecisionNode
            Root node of the built subtree
            
        TODO: Implement tree building algorithm
        1. Find best split by iterating through features and thresholds
        2. If good split found, recursively build left and right subtrees
        3. Otherwise, create leaf node
        """
        # TODO: Implement tree building
        pass

    def predict_value(self, x, tree=None):
        """
        Do a recursive search down the tree and make a prediction of the data sample
        by the value of the leaf that we end up at.
        
        Parameters:
        -----------
        x: array-like
            Single sample to predict
        tree: DecisionNode, optional
            Tree node to start from (defaults to root)
            
        Returns:
        --------
        prediction: float or int
            Predicted value
            
        TODO: Implement tree traversal for prediction
        1. If at leaf (tree.value is not None), return tree.value
        2. Otherwise, compare x[tree.feature_i] with tree.threshold
        3. Recursively traverse left or right subtree
        """
        # TODO: Implement prediction
        pass

    def predict(self, X):
        """
        Predict class or value for X.
        
        Parameters:
        -----------
        X: array-like, shape = [n_samples, n_features]
            Samples to predict
            
        Returns:
        --------
        y_pred: array-like, shape = [n_samples]
            Predicted values
            
        TODO: Apply predict_value to each sample in X
        """
        # TODO: Implement batch prediction
        pass


class RegressionTree(DecisionTree):
    """
    Regression tree for continuous target values.
    
    TODO: Implement regression-specific impurity and leaf value calculations
    """
    
    def _calculate_variance_reduction(self, y, y1, y2):
        """
        Calculate the variance reduction of a split.
        
        Parameters:
        -----------
        y: array-like
            Original targets
        y1: array-like  
            Left split targets
        y2: array-like
            Right split targets
            
        Returns:
        --------
        variance_reduction: float
            Reduction in variance
            
        TODO: Calculate variance reduction
        variance_reduction = var(y) - (|y1|/|y| * var(y1) + |y2|/|y| * var(y2))
        """
        # TODO: Implement variance reduction calculation
        pass
    
    def _mean_of_y(self, y):
        """
        Calculate mean of y values for leaf prediction.
        
        TODO: Return mean of y values
        """
        # TODO: Implement mean calculation
        pass
    
    def fit(self, X, y):
        """
        TODO: Set impurity and leaf value calculation functions, then call parent fit
        """
        # TODO: Set self._impurity_calculation = self._calculate_variance_reduction
        # TODO: Set self._leaf_value_calculation = self._mean_of_y
        # TODO: Call super().fit(X, y)
        pass


class ClassificationTree(DecisionTree):
    """
    Classification tree for discrete target values.
    
    TODO: Implement classification-specific impurity and leaf value calculations
    """
    
    def _calculate_information_gain(self, y, y1, y2):
        """
        Calculate the information gain of a split.
        
        Parameters:
        -----------
        y: array-like
            Original targets  
        y1: array-like
            Left split targets
        y2: array-like
            Right split targets
            
        Returns:
        --------
        information_gain: float
            Information gain from the split
            
        TODO: Calculate information gain
        information_gain = entropy(y) - (|y1|/|y| * entropy(y1) + |y2|/|y| * entropy(y2))
        """
        # TODO: Implement information gain calculation
        pass
    
    def _majority_vote(self, y):
        """
        Return the most common class label for leaf prediction.
        
        TODO: Return most frequent class in y
        """
        # TODO: Implement majority vote
        pass
    
    def fit(self, X, y):
        """
        TODO: Set impurity and leaf value calculation functions, then call parent fit
        """
        # TODO: Set self._impurity_calculation = self._calculate_information_gain  
        # TODO: Set self._leaf_value_calculation = self._majority_vote
        # TODO: Call super().fit(X, y)
        pass