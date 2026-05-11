import pytest
import torch
from unittest.mock import Mock, patch
from src.simulation.simulators.power_spectrum_simulator import PowerSpectrumSimulator


class TestPowerSpectrumSimulator:
    """Unit tests for PowerSpectrumSimulator."""
    
    def test_initialization(self):
        """Test PowerSpectrumSimulator initialization."""
        with patch('src.simulation.simulators.power_spectrum_simulator.CAMB') as mock_camb:
            mock_params = Mock()
            mock_camb.CAMBparams.return_value = mock_params
            
            simulator = PowerSpectrumSimulator(data_type="cmb")
            
            assert simulator.data_type == "cmb"
            mock_camb.CAMBparams.assert_called_once()
    
    def test_simulate_method(self):
        """Test the simulate method."""
        with patch('src.simulation.simulators.power_spectrum_simulator.CAMB') as mock_camb:
            # Mock CAMB setup
            mock_params = Mock()
            mock_camb.CAMBparams.return_value = mock_params
            
            # Mock results
            mock_results = Mock()
            mock_results.get_total_cl.return_value = torch.tensor([1000.0, 500.0, 250.0])
            mock_camb.get_results.return_value = mock_results
            
            simulator = PowerSpectrumSimulator()
            
            # Test parameters (simplified cosmological parameters)
            parameters = torch.tensor([70.0, 0.05, 0.25, 0.055, 2.1e-9, 0.96])
            result = simulator.simulate(parameters)
            
            assert isinstance(result, torch.Tensor)
            assert result.shape == (3,)  # TT, EE, TE power spectra
    
    def test_simulate_with_different_parameter_shapes(self):
        """Test simulate method with different parameter shapes."""
        with patch('src.simulation.simulators.power_spectrum_simulator.CAMB') as mock_camb:
            mock_params = Mock()
            mock_camb.CAMBparams.return_value = mock_params
            
            mock_results = Mock()
            mock_results.get_total_cl.return_value = torch.tensor([1000.0, 500.0, 250.0])
            mock_camb.get_results.return_value = mock_results
            
            simulator = PowerSpectrumSimulator()
            
            # Test with single parameter set
            single_params = torch.tensor([70.0, 0.05, 0.25, 0.055, 2.1e-9, 0.96])
            result1 = simulator.simulate(single_params)
            assert result1.shape == (3,)
            
            # Test with batch of parameters
            batch_params = torch.tensor([[70.0, 0.05, 0.25, 0.055, 2.1e-9, 0.96],
                                        [65.0, 0.04, 0.30, 0.060, 2.0e-9, 0.97]])
            result2 = simulator.simulate(batch_params)
            assert result2.shape == (2, 3)
    
    def test_parameter_validation(self):
        """Test parameter validation in simulate method."""
        with patch('src.simulation.simulators.power_spectrum_simulator.CAMB') as mock_camb:
            mock_params = Mock()
            mock_camb.CAMBparams.return_value = mock_params
            
            simulator = PowerSpectrumSimulator()
            
            # Test with wrong number of parameters
            wrong_params = torch.tensor([70.0, 0.05])  # Only 2 parameters instead of 6
            
            with pytest.raises(ValueError, match="Expected 6 parameters"):
                simulator.simulate(wrong_params)
            
            # Test with negative Hubble constant
            invalid_params = torch.tensor([-70.0, 0.05, 0.25, 0.055, 2.1e-9, 0.96])
            
            with pytest.raises(ValueError, match="Hubble constant must be positive"):
                simulator.simulate(invalid_params)
    
    def test_camb_parameter_setting(self):
        """Test that CAMB parameters are set correctly."""
        with patch('src.simulation.simulators.power_spectrum_simulator.CAMB') as mock_camb:
            mock_params = Mock()
            mock_camb.CAMBparams.return_value = mock_params
            
            mock_results = Mock()
            mock_results.get_total_cl.return_value = torch.tensor([1000.0, 500.0, 250.0])
            mock_camb.get_results.return_value = mock_results
            
            simulator = PowerSpectrumSimulator()
            
            parameters = torch.tensor([70.0, 0.05, 0.25, 0.055, 2.1e-9, 0.96])
            simulator.simulate(parameters)
            
            # Verify that CAMB parameters were set
            assert mock_params.set_cosmology.called
            assert mock_params.InitPower.set_params.called
            assert mock_camb.get_results.called
    
    def test_data_type_handling(self):
        """Test different data types."""
        with patch('src.simulation.simulators.power_spectrum_simulator.CAMB') as mock_camb:
            mock_params = Mock()
            mock_camb.CAMBparams.return_value = mock_params
            
            # Test CMB data type
            cmb_simulator = PowerSpectrumSimulator(data_type="cmb")
            assert cmb_simulator.data_type == "cmb"
            
            # Test LSS data type
            lss_simulator = PowerSpectrumSimulator(data_type="lss")
            assert lss_simulator.data_type == "lss"
    
    def test_error_handling(self):
        """Test error handling in CAMB calls."""
        with patch('src.simulation.simulators.power_spectrum_simulator.CAMB') as mock_camb:
            mock_params = Mock()
            mock_camb.CAMBparams.return_value = mock_params
            
            # Mock CAMB to raise an exception
            mock_camb.get_results.side_effect = Exception("CAMB calculation failed")
            
            simulator = PowerSpectrumSimulator()
            parameters = torch.tensor([70.0, 0.05, 0.25, 0.055, 2.1e-9, 0.96])
            
            with pytest.raises(Exception, match="CAMB calculation failed"):
                simulator.simulate(parameters)
