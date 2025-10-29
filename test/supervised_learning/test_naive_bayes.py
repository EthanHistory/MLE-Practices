"""
Additional test cases for Naive Bayes and Perceptron implementations.
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import original implementations
from mlfromscratch.supervised_learning.naive_bayes import NaiveBayes as OriginalNaiveBayes
from mlfromscratch.supervised_learning.perceptron import Perceptron as OriginalPerceptron

# Import practice implementations
try:
    from practice.supervised_learning.naive_bayes import NaiveBayes as PracticeNaiveBayes
    from practice.supervised_learning.perceptron import Perceptron as PracticePerceptron
except ImportError:
    PracticeNaiveBayes = None
    PracticePerceptron = None


class TestNaiveBayes:
    """Test cases for NaiveBayes"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample classification data"""
        np.random.seed(42)
        X, y = make_classification(n_samples=200, n_features=4, n_redundant=0, 
                                   n_informative=3, n_clusters_per_class=1, random_state=42)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        return X_train, X_test, y_train, y_test
    
    def test_practice_implementation_exists(self):
        """Test that practice implementation exists and can be imported"""
        assert PracticeNaiveBayes is not None, "Practice NaiveBayes implementation not found!"
    
    @pytest.mark.skipif(PracticeNaiveBayes is None, 
                        reason="Practice implementation not available")
    def test_naive_bayes_predict(self, sample_data):
        """Test NaiveBayes fitting and prediction"""
        X_train, X_test, y_train, y_test = sample_data
        
        # Original implementation
        original = OriginalNaiveBayes()
        original.fit(X_train, y_train)
        original_pred = original.predict(X_test)
        
        # Practice implementation
        practice = PracticeNaiveBayes()
        practice.fit(X_train, y_train)
        practice_pred = practice.predict(X_test)
        
        # Calculate accuracies
        original_accuracy = np.mean(original_pred == y_test)
        practice_accuracy = np.mean(practice_pred == y_test)
        
        print(f"Original accuracy: {original_accuracy:.4f}")
        print(f"Practice accuracy: {practice_accuracy:.4f}")
        
        # Check that accuracies are close (within 15% tolerance)
        tolerance = 0.15
        assert abs(original_accuracy - practice_accuracy) <= tolerance, \
            f"Accuracy difference too large: {abs(original_accuracy - practice_accuracy):.4f} > {tolerance}"
        
        # Check that both achieve reasonable accuracy (> 0.7)
        assert original_accuracy > 0.7, f"Original accuracy too low: {original_accuracy:.4f}"
        assert practice_accuracy > 0.7, f"Practice accuracy too low: {practice_accuracy:.4f}"


if __name__ == "__main__":
    pytest.main([__file__])