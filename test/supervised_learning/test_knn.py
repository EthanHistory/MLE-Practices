"""
Test cases for K-Nearest Neighbors implementation.
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
from mlfromscratch.supervised_learning.k_nearest_neighbors import KNN as OriginalKNN

# Import practice implementation
try:
    from practice.supervised_learning.k_nearest_neighbors import KNN as PracticeKNN
except ImportError:
    PracticeKNN = None


class TestKNN:
    """Test cases for KNN"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample classification data"""
        np.random.seed(42)
        X, y = make_classification(n_samples=150, n_features=2, n_redundant=0, 
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
        assert PracticeKNN is not None, "Practice KNN implementation not found!"
    
    @pytest.mark.skipif(PracticeKNN is None, 
                        reason="Practice implementation not available")
    def test_initialization(self):
        """Test that both implementations can be initialized"""
        original = OriginalKNN(k=5)
        practice = PracticeKNN(k=5)
        
        assert original is not None
        assert practice is not None
        assert original.k == 5
        assert practice.k == 5
    
    @pytest.mark.skipif(PracticeKNN is None, 
                        reason="Practice implementation not available")
    def test_predict(self, sample_data):
        """Test that both implementations can predict and produce similar results"""
        X_train, X_test, y_train, y_test = sample_data
        
        # Test with different k values
        for k in [3, 5, 7]:
            # Original implementation
            original = OriginalKNN(k=k)
            original_pred = original.predict(X_test, X_train, y_train)
            
            # Practice implementation  
            practice = PracticeKNN(k=k)
            practice_pred = practice.predict(X_test, X_train, y_train)
            
            # Calculate accuracies
            original_accuracy = np.mean(original_pred == y_test)
            practice_accuracy = np.mean(practice_pred == y_test)
            
            print(f"k={k} - Original accuracy: {original_accuracy:.4f}")
            print(f"k={k} - Practice accuracy: {practice_accuracy:.4f}")
            
            # Check that accuracies are close (within 5% tolerance)
            tolerance = 0.05
            assert abs(original_accuracy - practice_accuracy) <= tolerance, \
                f"k={k}: Accuracy difference too large: {abs(original_accuracy - practice_accuracy):.4f} > {tolerance}"
            
            # Check that both achieve reasonable accuracy (> 0.75)
            assert original_accuracy > 0.75, f"k={k}: Original accuracy too low: {original_accuracy:.4f}"
            assert practice_accuracy > 0.75, f"k={k}: Practice accuracy too low: {practice_accuracy:.4f}"
    
    @pytest.mark.skipif(PracticeKNN is None, 
                        reason="Practice implementation not available")
    def test_exact_match_small_dataset(self):
        """Test that both implementations produce identical results on a small dataset"""
        # Create a very small, simple dataset
        np.random.seed(0)
        X_train = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]])
        y_train = np.array([0, 0, 0, 1, 1, 1])
        X_test = np.array([[2, 2], [7, 6]])
        
        # Original implementation
        original = OriginalKNN(k=3)
        original_pred = original.predict(X_test, X_train, y_train)
        
        # Practice implementation  
        practice = PracticeKNN(k=3)
        practice_pred = practice.predict(X_test, X_train, y_train)
        
        # Results should be identical
        np.testing.assert_array_equal(original_pred, practice_pred, 
                                      "Predictions should be identical on small dataset")


if __name__ == "__main__":
    pytest.main([__file__])