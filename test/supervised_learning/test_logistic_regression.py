"""
Test cases for comparing practice implementations with original implementations.
Tests LogisticRegression implementation.
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

# Import original implementation
from mlfromscratch.supervised_learning.logistic_regression import LogisticRegression as OriginalLogisticRegression

# Import practice implementation
try:
    from practice.supervised_learning.logistic_regression import LogisticRegression as PracticeLogisticRegression
except ImportError:
    PracticeLogisticRegression = None


class TestLogisticRegression:
    """Test cases for LogisticRegression"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample classification data"""
        np.random.seed(42)
        X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                                   n_informative=2, n_clusters_per_class=1, random_state=42)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Standardize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def test_practice_implementation_exists(self):
        """Test that practice implementation exists and can be imported"""
        assert PracticeLogisticRegression is not None, "Practice LogisticRegression implementation not found!"
    
    @pytest.mark.skipif(PracticeLogisticRegression is None, 
                        reason="Practice implementation not available")
    def test_initialization(self):
        """Test that both implementations can be initialized"""
        original = OriginalLogisticRegression()
        practice = PracticeLogisticRegression()
        
        assert original is not None
        assert practice is not None
    
    @pytest.mark.skipif(PracticeLogisticRegression is None, 
                        reason="Practice implementation not available")
    def test_fit_and_predict(self, sample_data):
        """Test that both implementations can fit and predict"""
        X_train, X_test, y_train, y_test = sample_data
        
        # Original implementation
        original = OriginalLogisticRegression(learning_rate=0.1, gradient_descent=True)
        original.fit(X_train, y_train, n_iterations=1000)
        original_pred = original.predict(X_test)
        
        # Practice implementation  
        practice = PracticeLogisticRegression(learning_rate=0.1, gradient_descent=True)
        practice.fit(X_train, y_train, n_iterations=1000)
        practice_pred = practice.predict(X_test)
        
        # Calculate accuracies
        original_accuracy = np.mean(original_pred == y_test)
        practice_accuracy = np.mean(practice_pred == y_test)
        
        print(f"Original accuracy: {original_accuracy:.4f}")
        print(f"Practice accuracy: {practice_accuracy:.4f}")
        
        # Check that accuracies are close (within 10% tolerance)
        tolerance = 0.1
        assert abs(original_accuracy - practice_accuracy) <= tolerance, \
            f"Accuracy difference too large: {abs(original_accuracy - practice_accuracy):.4f} > {tolerance}"
        
        # Check that both achieve reasonable accuracy (> 0.7)
        assert original_accuracy > 0.7, f"Original accuracy too low: {original_accuracy:.4f}"
        assert practice_accuracy > 0.7, f"Practice accuracy too low: {practice_accuracy:.4f}"
    
    @pytest.mark.skipif(PracticeLogisticRegression is None, 
                        reason="Practice implementation not available")
    def test_parameter_convergence(self, sample_data):
        """Test that learned parameters are similar between implementations"""
        X_train, X_test, y_train, y_test = sample_data
        
        # Use same random seed for both
        np.random.seed(42)
        original = OriginalLogisticRegression(learning_rate=0.01, gradient_descent=True)
        original.fit(X_train, y_train, n_iterations=2000)
        
        np.random.seed(42)
        practice = PracticeLogisticRegression(learning_rate=0.01, gradient_descent=True)
        practice.fit(X_train, y_train, n_iterations=2000)
        
        # Compare learned parameters (allow some tolerance due to different implementations)
        if hasattr(original, 'param') and hasattr(practice, 'param'):
            param_diff = np.linalg.norm(original.param - practice.param)
            print(f"Parameter difference (L2 norm): {param_diff:.4f}")
            
            # Parameters should be reasonably close
            tolerance = 2.0  # Generous tolerance for parameter differences
            assert param_diff <= tolerance, \
                f"Parameters too different: {param_diff:.4f} > {tolerance}"


if __name__ == "__main__":
    pytest.main([__file__])