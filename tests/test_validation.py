import pytest
import numpy as np
from pynomial.intervals import _check_inputs


@pytest.mark.validation
class TestCheckInputsValidation:
    """Test suite for _check_inputs function validation."""

    def test_none_x_raises_error(self):
        """Test that None value for x raises ValueError."""
        with pytest.raises(ValueError, match="x and n must not be None"):
            _check_inputs(None, 10)

    def test_none_n_raises_error(self):
        """Test that None value for n raises ValueError."""
        with pytest.raises(ValueError, match="x and n must not be None"):
            _check_inputs(5, None)

    def test_both_none_raises_error(self):
        """Test that None values for both x and n raise ValueError."""
        with pytest.raises(ValueError, match="x and n must not be None"):
            _check_inputs(None, None)

    def test_x_negative_raises_error(self):
        """Test that negative x values raise ValueError."""
        with pytest.raises(ValueError, match="x must be between 0 and n"):
            _check_inputs(-1, 10)

    def test_x_greater_than_n_raises_error(self):
        """Test that x > n raises ValueError."""
        with pytest.raises(ValueError, match="x must be between 0 and n"):
            _check_inputs(15, 10)

    def test_n_negative_raises_error(self):
        """Test that negative n values raise ValueError."""
        with pytest.raises(ValueError, match="n must be greater than 0"):
            _check_inputs(5, -1)

    def test_confidence_level_negative_raises_error(self):
        """Test that negative confidence_level raises ValueError."""
        with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
            _check_inputs(5, 10, confidence_level=-0.1)

    def test_confidence_level_greater_than_one_raises_error(self):
        """Test that confidence_level > 1 raises ValueError."""
        with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
            _check_inputs(5, 10, confidence_level=1.1)

    def test_broadcasting_incompatible_shapes_raises_error(self):
        """Test that incompatible broadcasting shapes raise ValueError."""
        x = np.array([1, 2, 3])  # shape (3,)
        n = np.array([[10, 20], [30, 40]])  # shape (2, 2)
        confidence_level = np.array([0.95, 0.99, 0.90, 0.85])  # shape (4,)
        
        with pytest.raises(ValueError, match="x, n, and confidence_level must be broadcastable to the same shape"):
            _check_inputs(x, n, confidence_level)

    def test_array_x_negative_values_raises_error(self):
        """Test that arrays with negative x values raise ValueError."""
        x = np.array([1, -2, 3])
        n = np.array([10, 10, 10])
        
        with pytest.raises(ValueError, match="x must be between 0 and n"):
            _check_inputs(x, n)

    def test_array_x_greater_than_n_raises_error(self):
        """Test that arrays with x > n raise ValueError."""
        x = np.array([1, 15, 3])
        n = np.array([10, 10, 10])
        
        with pytest.raises(ValueError, match="x must be between 0 and n"):
            _check_inputs(x, n)

    def test_array_n_negative_values_raises_error(self):
        """Test that arrays with negative n values raise ValueError."""
        x = np.array([1, 2, 3])
        n = np.array([10, -5, 10])
        
        with pytest.raises(ValueError, match="n must be greater than 0"):
            _check_inputs(x, n)

    def test_array_confidence_level_out_of_bounds_raises_error(self):
        """Test that arrays with confidence_level out of bounds raise ValueError."""
        x = 5
        n = 10
        confidence_level = np.array([0.95, 1.1, 0.90])
        
        with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
            _check_inputs(x, n, confidence_level)

    # Valid input tests
    def test_valid_scalar_inputs(self):
        """Test that valid scalar inputs pass without error."""
        # Should not raise any exception
        _check_inputs(5, 10, 0.95)

    def test_valid_x_equals_n(self):
        """Test that x == n is valid."""
        # Should not raise any exception
        _check_inputs(10, 10, 0.95)

    def test_valid_x_equals_zero(self):
        """Test that x == 0 is valid."""
        # Should not raise any exception
        _check_inputs(0, 10, 0.95)

    def test_valid_confidence_level_boundaries(self):
        """Test that confidence_level at boundaries (0 and 1) is valid."""
        # Should not raise any exception
        _check_inputs(5, 10, 0.0)
        _check_inputs(5, 10, 1.0)

    def test_valid_array_inputs(self):
        """Test that valid array inputs pass without error."""
        x = np.array([1, 2, 3])
        n = np.array([10, 10, 10])
        confidence_level = np.array([0.95, 0.99, 0.90])
        
        # Should not raise any exception
        _check_inputs(x, n, confidence_level)

    def test_valid_broadcastable_shapes(self):
        """Test that valid broadcastable shapes pass without error."""
        x = np.array([1, 2])  # shape (2,)
        n = np.array([[10], [20]])  # shape (2, 1)
        confidence_level = 0.95  # scalar
        
        # Should not raise any exception
        _check_inputs(x, n, confidence_level)

    def test_valid_mixed_scalar_array(self):
        """Test that mixed scalar and array inputs pass when valid."""
        x = 5  # scalar
        n = np.array([10, 15, 20])  # array
        confidence_level = 0.95  # scalar
        
        # Should not raise any exception
        _check_inputs(x, n, confidence_level)

    def test_default_confidence_level(self):
        """Test that default confidence_level works correctly."""
        # Should not raise any exception when confidence_level is not provided
        _check_inputs(5, 10)

    # Test precedence order for multiple validation errors
    def test_precedence_x_negative_over_n_negative(self):
        """Test that x < 0 error takes precedence over n < 0 error."""
        with pytest.raises(ValueError, match="x must be between 0 and n"):
            _check_inputs(-1, -5)  # Both x and n are negative, x error should come first

    def test_precedence_x_negative_over_x_greater_than_n(self):
        """Test that x < 0 error takes precedence over x > n error."""
        with pytest.raises(ValueError, match="x must be between 0 and n"):
            _check_inputs(-1, 5)  # x is negative, should trigger first check

    def test_precedence_n_negative_over_x_greater_than_n(self):
        """Test that n < 0 error takes precedence over x > n error."""
        with pytest.raises(ValueError, match="n must be greater than 0"):
            _check_inputs(10, -5)  # n is negative and x > n, n error should come first

    def test_precedence_with_arrays_x_negative_first(self):
        """Test precedence with arrays where x has negative values."""
        x = np.array([-1, 2, 15])  # negative, valid, greater than n
        n = np.array([-5, 10, 10])  # negative, valid, valid
        
        # Should catch x < 0 first
        with pytest.raises(ValueError, match="x must be between 0 and n"):
            _check_inputs(x, n)

    def test_precedence_with_arrays_n_negative_second(self):
        """Test precedence with arrays where n has negative values but x is valid."""
        x = np.array([1, 2, 15])  # valid, valid, greater than n
        n = np.array([-5, 10, 10])  # negative, valid, valid
        
        # Should catch n < 0 second (no x < 0 to catch first)
        with pytest.raises(ValueError, match="n must be greater than 0"):
            _check_inputs(x, n)

    def test_precedence_with_arrays_x_greater_than_n_third(self):
        """Test precedence with arrays where only x > n violation exists."""
        x = np.array([1, 2, 15])  # valid, valid, greater than n
        n = np.array([5, 10, 10])  # valid, valid, valid
        
        # Should catch x > n third (no x < 0 or n < 0 to catch first)
        with pytest.raises(ValueError, match="x must be between 0 and n"):
            _check_inputs(x, n)
