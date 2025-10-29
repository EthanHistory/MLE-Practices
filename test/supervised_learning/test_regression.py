"""
Test cases for Linear Regression and other regression models.
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Import original implementations
from mlfromscratch.supervised_learning.regression import (
    LinearRegression as OriginalLinearRegression,
    LassoRegression as OriginalLassoRegression,
    RidgeRegression as OriginalRidgeRegression,
    PolynomialRegression as OriginalPolynomialRegression
)

# Import practice implementations
try:
    from practice.supervised_learning.regression import (
        LinearRegression as PracticeLinearRegression,
        LassoRegression as PracticeLassoRegression, 
        RidgeRegression as PracticeRidgeRegression,
        PolynomialRegression as PracticePolynomialRegression
    )
except ImportError:
    PracticeLinearRegression = None
    PracticeLassoRegression = None
    PracticeRidgeRegression = None
    PracticePolynomialRegression = None


class TestLinearRegression:
    """Test cases for LinearRegression"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample regression data"""
        np.random.seed(42)
        X, y = make_regression(n_samples=100, n_features=2, noise=10, random_state=42)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Standardize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def test_practice_implementation_exists(self):
        """Test that practice implementation exists and can be imported"""
        assert PracticeLinearRegression is not None, "Practice LinearRegression implementation not found!"
    
    @pytest.mark.skipif(PracticeLinearRegression is None, 
                        reason="Practice implementation not available")
    def test_linear_regression_gradient_descent(self, sample_data):
        """Test LinearRegression with gradient descent"""
        X_train, X_test, y_train, y_test = sample_data
        
        # Original implementation
        original = OriginalLinearRegression(n_iterations=1000, learning_rate=0.01, gradient_descent=True)
        original.fit(X_train, y_train)
        original_pred = original.predict(X_test)
        
        # Practice implementation
        practice = PracticeLinearRegression(n_iterations=1000, learning_rate=0.01, gradient_descent=True)
        practice.fit(X_train, y_train)
        practice_pred = practice.predict(X_test)
        
        # Calculate metrics
        original_mse = mean_squared_error(y_test, original_pred)
        practice_mse = mean_squared_error(y_test, practice_pred)
        
        original_r2 = r2_score(y_test, original_pred)
        practice_r2 = r2_score(y_test, practice_pred)
        
        print(f"Original MSE: {original_mse:.4f}, R²: {original_r2:.4f}")
        print(f"Practice MSE: {practice_mse:.4f}, R²: {practice_r2:.4f}")
        
        # Check that MSE are close (within 20% tolerance)
        mse_tolerance = 0.2 * original_mse
        assert abs(original_mse - practice_mse) <= mse_tolerance, \
            f"MSE difference too large: {abs(original_mse - practice_mse):.4f} > {mse_tolerance:.4f}"
        
        # Check that both achieve reasonable R² (> 0.8)
        assert original_r2 > 0.8, f"Original R² too low: {original_r2:.4f}"
        assert practice_r2 > 0.8, f"Practice R² too low: {practice_r2:.4f}"
    
    @pytest.mark.skipif(PracticeLinearRegression is None, 
                        reason="Practice implementation not available")
    def test_linear_regression_least_squares(self, sample_data):
        """Test LinearRegression with least squares"""
        X_train, X_test, y_train, y_test = sample_data
        
        # Original implementation
        original = OriginalLinearRegression(gradient_descent=False)
        original.fit(X_train, y_train)
        original_pred = original.predict(X_test)
        
        # Practice implementation
        practice = PracticeLinearRegression(gradient_descent=False)
        practice.fit(X_train, y_train)
        practice_pred = practice.predict(X_test)
        
        # Calculate metrics
        original_mse = mean_squared_error(y_test, original_pred)
        practice_mse = mean_squared_error(y_test, practice_pred)
        
        print(f"Least Squares - Original MSE: {original_mse:.4f}")
        print(f"Least Squares - Practice MSE: {practice_mse:.4f}")
        
        # With least squares, results should be very close
        mse_tolerance = 0.01 * original_mse
        assert abs(original_mse - practice_mse) <= mse_tolerance, \
            f"MSE difference too large for least squares: {abs(original_mse - practice_mse):.4f} > {mse_tolerance:.4f}"


class TestPolynomialRegression:
    """Test cases for PolynomialRegression"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample polynomial data"""
        np.random.seed(42)
        X = np.linspace(-2, 2, 80).reshape(-1, 1)
        y = 0.5 * X.ravel() ** 2 + 0.3 * X.ravel() + 0.1 + np.random.normal(0, 0.1, X.shape[0])
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        return X_train, X_test, y_train, y_test
    
    @pytest.mark.skipif(PracticePolynomialRegression is None, 
                        reason="Practice implementation not available")
    def test_polynomial_regression(self, sample_data):
        """Test PolynomialRegression"""
        X_train, X_test, y_train, y_test = sample_data
        
        degree = 2
        
        # Original implementation
        original = OriginalPolynomialRegression(degree=degree, n_iterations=1000, learning_rate=0.01)
        original.fit(X_train, y_train)
        original_pred = original.predict(X_test)
        
        # Practice implementation
        practice = PracticePolynomialRegression(degree=degree, n_iterations=1000, learning_rate=0.01)
        practice.fit(X_train, y_train)
        practice_pred = practice.predict(X_test)
        
        # Calculate metrics
        original_mse = mean_squared_error(y_test, original_pred)
        practice_mse = mean_squared_error(y_test, practice_pred)
        
        original_r2 = r2_score(y_test, original_pred)
        practice_r2 = r2_score(y_test, practice_pred)
        
        print(f"Polynomial - Original MSE: {original_mse:.4f}, R²: {original_r2:.4f}")
        print(f"Polynomial - Practice MSE: {practice_mse:.4f}, R²: {practice_r2:.4f}")
        
        # Check that MSE are close (within 30% tolerance for polynomial)
        mse_tolerance = 0.3 * original_mse
        assert abs(original_mse - practice_mse) <= mse_tolerance, \
            f"MSE difference too large: {abs(original_mse - practice_mse):.4f} > {mse_tolerance:.4f}"
        
        # Check that both achieve reasonable R² (> 0.7)
        assert original_r2 > 0.7, f"Original R² too low: {original_r2:.4f}"
        assert practice_r2 > 0.7, f"Practice R² too low: {practice_r2:.4f}"


if __name__ == "__main__":
    pytest.main([__file__])