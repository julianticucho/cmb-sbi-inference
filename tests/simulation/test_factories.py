import pytest
from unittest.mock import Mock, patch
from src.simulation.factories.prior_factory import PriorFactory
from src.simulation.factories.simulator_factory import SimulatorFactory


class TestPriorFactory:
    """Unit tests for PriorFactory."""
    
    def test_create_standard_cosmology_prior(self):
        """Test creation of standard cosmology prior."""
        # This test would need to be implemented based on the actual PriorFactory
        # For now, we'll create a placeholder that shows the structure
        with patch('src.simulation.factories.prior_factory.StandardCosmologyPrior') as mock_prior:
            mock_instance = Mock()
            mock_prior.return_value = mock_instance
            
            factory = PriorFactory()
            result = factory.create("standard_cosmology")
            
            mock_prior.assert_called_once()
            assert result == mock_instance
    
    def test_create_unknown_prior_raises_error(self):
        """Test that creating unknown prior type raises error."""
        factory = PriorFactory()
        
        with pytest.raises(ValueError, match="Unknown prior type"):
            factory.create("unknown_prior_type")
    
    def test_get_available_prior_types(self):
        """Test getting list of available prior types."""
        factory = PriorFactory()
        available_types = factory.get_available_types()
        
        assert isinstance(available_types, list)
        assert "standard_cosmology" in available_types


class TestSimulatorFactory:
    """Unit tests for SimulatorFactory."""
    
    def test_create_power_spectrum_simulator(self):
        """Test creation of power spectrum simulator."""
        with patch('src.simulation.factories.simulator_factory.PowerSpectrumSimulator') as mock_simulator:
            mock_instance = Mock()
            mock_simulator.return_value = mock_instance
            
            factory = SimulatorFactory()
            result = factory.create("power_spectrum")
            
            mock_simulator.assert_called_once()
            assert result == mock_instance
    
    def test_create_simulator_with_config(self):
        """Test creating simulator with configuration."""
        config = {"data_type": "cmb", "resolution": 512}
        
        with patch('src.simulation.factories.simulator_factory.PowerSpectrumSimulator') as mock_simulator:
            mock_instance = Mock()
            mock_simulator.return_value = mock_instance
            
            factory = SimulatorFactory()
            result = factory.create("power_spectrum", config)
            
            mock_simulator.assert_called_once_with(**config)
            assert result == mock_instance
    
    def test_create_unknown_simulator_raises_error(self):
        """Test that creating unknown simulator type raises error."""
        factory = SimulatorFactory()
        
        with pytest.raises(ValueError, match="Unknown simulator type"):
            factory.create("unknown_simulator")
    
    def test_get_available_simulator_types(self):
        """Test getting list of available simulator types."""
        factory = SimulatorFactory()
        available_types = factory.get_available_types()
        
        assert isinstance(available_types, list)
        assert "power_spectrum" in available_types
