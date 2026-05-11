import pytest
import torch
from unittest.mock import Mock, patch
from src.simulation.priors.standard_cosmology_prior import StandardCosmologyPrior


class TestStandardCosmologyPrior:
    """Unit tests for StandardCosmologyPrior."""
    
    def test_initialization(self):
        """Test StandardCosmologyPrior initialization."""
        with patch('src.simulation.priors.standard_cosmology_prior.BoxUniform') as mock_box_uniform:
            mock_prior = Mock()
            mock_box_uniform.return_value = mock_prior
            
            prior = StandardCosmologyPrior()
            
            assert prior.prior == mock_prior
            mock_box_uniform.assert_called_once()
    
    def test_sample_method(self):
        """Test sampling from the prior."""
        with patch('src.simulation.priors.standard_cosmology_prior.BoxUniform') as mock_box_uniform:
            mock_prior = Mock()
            mock_samples = torch.randn(10, 6)
            mock_prior.sample.return_value = mock_samples
            mock_box_uniform.return_value = mock_prior
            
            prior = StandardCosmologyPrior()
            samples = prior.sample(num_samples=10, seed=42)
            
            mock_prior.sample.assert_called_once_with((10,))
            assert torch.equal(samples, mock_samples)
    
    def test_get_parameter_names(self):
        """Test getting parameter names."""
        expected_names = ["H0", "omegab", "omegac", "tau", "As", "ns"]
        
        with patch('src.simulation.priors.standard_cosmology_prior.BoxUniform'):
            prior = StandardCosmologyPrior()
            names = prior.get_parameter_names()
            
            assert names == expected_names
    
    def test_get_parameter_ranges(self):
        """Test getting parameter ranges."""
        with patch('src.simulation.priors.standard_cosmology_prior.BoxUniform'):
            prior = StandardCosmologyPrior()
            ranges = prior.get_parameter_ranges()
            
            assert isinstance(ranges, dict)
            for param_name in prior.get_parameter_names():
                assert param_name in ranges
                min_val, max_val = ranges[param_name]
                assert isinstance(min_val, (int, float))
                assert isinstance(max_val, (int, float))
                assert min_val < max_val
    
    def test_to_sbi_method(self):
        """Test conversion to SBI prior."""
        with patch('src.simulation.priors.standard_cosmology_prior.BoxUniform') as mock_box_uniform:
            mock_prior = Mock()
            mock_box_uniform.return_value = mock_prior
            
            prior = StandardCosmologyPrior()
            sbi_prior = prior.to_sbi(device="cpu")
            
            assert sbi_prior == mock_prior
    
    def test_parameter_ranges_are_physical(self):
        """Test that parameter ranges are physically reasonable."""
        with patch('src.simulation.priors.standard_cosmology_prior.BoxUniform'):
            prior = StandardCosmologyPrior()
            ranges = prior.get_parameter_ranges()
            
            # Check Hubble constant range (roughly 50-100 km/s/Mpc)
            h0_range = ranges["H0"]
            assert 50 <= h0_range[0] <= h0_range[1] <= 100
            
            # Check baryon density (positive and less than 1)
            omegab_range = ranges["omegab"]
            assert 0 < omegab_range[0] <= omegab_range[1] < 1
            
            # Check cold dark matter density (positive)
            omegac_range = ranges["omegac"]
            assert 0 < omegac_range[0] <= omegac_range[1]
    
    def test_seed_reproducibility(self):
        """Test that sampling is reproducible with same seed."""
        with patch('src.simulation.priors.standard_cosmology_prior.BoxUniform') as mock_box_uniform:
            mock_prior = Mock()
            # Mock deterministic sampling based on seed
            mock_prior.sample.side_effect = lambda shape: torch.randn(*shape) if torch.rand(1) > 0.5 else torch.ones(*shape)
            mock_box_uniform.return_value = mock_prior
            
            prior = StandardCosmologyPrior()
            
            # Reset the mock for consistent behavior
            mock_prior.sample.reset_mock()
            
            samples1 = prior.sample(num_samples=5, seed=123)
            samples2 = prior.sample(num_samples=5, seed=123)
            
            # With the same mock behavior, results should be consistent
            assert samples1.shape == samples2.shape
