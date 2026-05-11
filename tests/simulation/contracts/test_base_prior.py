import pytest
import torch
from unittest.mock import Mock
from src.simulation.contracts.base_prior import BasePrior


class MockPrior(BasePrior):
    """Mock implementation of BasePrior for testing."""
    
    def __init__(self, param_names=None, param_ranges=None):
        self._param_names = param_names or ["param1", "param2"]
        self._param_ranges = param_ranges or {"param1": (0.0, 1.0), "param2": (-1.0, 1.0)}
    
    def sample(self, num_samples: int, seed: int = None) -> torch.Tensor:
        """Mock sample method."""
        if seed is not None:
            torch.manual_seed(seed)
        return torch.randn(num_samples, len(self._param_names))
    
    def get_parameter_names(self) -> list[str]:
        """Mock parameter names method."""
        return self._param_names
    
    def get_parameter_ranges(self) -> dict:
        """Mock parameter ranges method."""
        return self._param_ranges
    
    def to_sbi(self, device: str = "cpu"):
        """Mock sbi conversion method."""
        mock_sbi_prior = Mock()
        return mock_sbi_prior


class TestBasePrior:
    """Unit tests for BasePrior abstract class."""
    
    def test_sample_method(self):
        """Test sample method implementation."""
        prior = MockPrior()
        samples = prior.sample(num_samples=10, seed=42)
        
        assert samples.shape == (10, 2)
        assert isinstance(samples, torch.Tensor)
    
    def test_sample_without_seed(self):
        """Test sample method without seed."""
        prior = MockPrior()
        samples = prior.sample(num_samples=5)
        
        assert samples.shape == (5, 2)
        assert isinstance(samples, torch.Tensor)
    
    def test_get_parameter_names(self):
        """Test get_parameter_names method."""
        expected_names = ["param1", "param2", "param3"]
        prior = MockPrior(param_names=expected_names)
        
        names = prior.get_parameter_names()
        assert names == expected_names
    
    def test_get_parameter_ranges(self):
        """Test get_parameter_ranges method."""
        expected_ranges = {"param1": (0.0, 1.0), "param2": (-2.0, 2.0)}
        prior = MockPrior(param_ranges=expected_ranges)
        
        ranges = prior.get_parameter_ranges()
        assert ranges == expected_ranges
    
    def test_to_sbi(self):
        """Test to_sbi method."""
        prior = MockPrior()
        sbi_prior = prior.to_sbi(device="cpu")
        
        # Verify that a mock object is returned
        assert sbi_prior is not None
    
    def test_to_sbi_with_gpu_device(self):
        """Test to_sbi method with GPU device."""
        prior = MockPrior()
        sbi_prior = prior.to_sbi(device="cuda")
        
        assert sbi_prior is not None
    
    def test_abstract_method_enforcement(self):
        """Test that BasePrior cannot be instantiated without implementing abstract methods."""
        with pytest.raises(TypeError):
            BasePrior()
    
    def test_parameter_ranges_structure(self):
        """Test that parameter ranges have correct structure."""
        prior = MockPrior()
        ranges = prior.get_parameter_ranges()
        
        assert isinstance(ranges, dict)
        for param_name, (min_val, max_val) in ranges.items():
            assert isinstance(param_name, str)
            assert isinstance(min_val, (int, float))
            assert isinstance(max_val, (int, float))
            assert min_val < max_val
    
    def test_sample_shape_consistency(self):
        """Test that sample shape is consistent with parameter names."""
        param_names = ["p1", "p2", "p3", "p4"]
        prior = MockPrior(param_names=param_names)
        
        samples = prior.sample(num_samples=100)
        assert samples.shape == (100, len(param_names))
    
    def test_seed_reproducibility(self):
        """Test that using the same seed produces reproducible results."""
        prior = MockPrior()
        
        samples1 = prior.sample(num_samples=10, seed=123)
        samples2 = prior.sample(num_samples=10, seed=123)
        
        assert torch.equal(samples1, samples2)
    
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        prior = MockPrior()
        
        samples1 = prior.sample(num_samples=10, seed=123)
        samples2 = prior.sample(num_samples=10, seed=456)
        
        assert not torch.equal(samples1, samples2)
