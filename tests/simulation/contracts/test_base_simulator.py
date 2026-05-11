import pytest
import torch
from unittest.mock import Mock, patch
from src.simulation.contracts.base_simulator import BaseSimulator


class MockSimulator(BaseSimulator):
    """Mock implementation of BaseSimulator for testing."""
    
    def __init__(self, data_type: str = "test"):
        super().__init__(data_type)
    
    def simulate(self, parameters: torch.Tensor) -> torch.Tensor:
        """Mock simulate method that returns parameters squared."""
        return parameters ** 2


class TestBaseSimulator:
    """Unit tests for BaseSimulator abstract class."""
    
    def test_initialization(self):
        """Test BaseSimulator initialization."""
        simulator = MockSimulator("test_data")
        assert simulator.data_type == "test_data"
    
    def test_simulate_method(self):
        """Test concrete simulate method implementation."""
        simulator = MockSimulator()
        parameters = torch.tensor([1.0, 2.0, 3.0])
        result = simulator.simulate(parameters)
        
        expected = parameters ** 2
        assert torch.equal(result, expected)
    
    @patch('src.simulation.contracts.base_simulator.process_prior')
    @patch('src.simulation.contracts.base_simulator.process_simulator')
    @patch('src.simulation.contracts.base_simulator.simulate_for_sbi')
    def test_simulate_batch(self, mock_simulate_for_sbi, mock_process_simulator, mock_process_prior):
        """Test batch simulation with mocked dependencies."""
        # Setup mocks
        mock_prior = Mock()
        mock_processed_prior = Mock()
        mock_prior_returns_numpy = False
        mock_process_prior.return_value = (mock_processed_prior, None, mock_prior_returns_numpy)
        
        mock_simulator_wrapper = Mock()
        mock_process_simulator.return_value = mock_simulator_wrapper
        
        mock_theta = torch.randn(10, 3)
        mock_x = torch.randn(10, 2)
        mock_simulate_for_sbi.return_value = (mock_theta, mock_x)
        
        # Test
        simulator = MockSimulator()
        result_theta, result_x = simulator.simulate_batch(
            num_simulations=10,
            prior=mock_prior,
            seed=42,
            num_workers=2
        )
        
        # Assertions
        mock_process_prior.assert_called_once_with(mock_prior)
        mock_process_simulator.assert_called_once()
        mock_simulate_for_sbi.assert_called_once_with(
            mock_simulator_wrapper,
            proposal=mock_processed_prior,
            num_simulations=10,
            num_workers=2,
            seed=42
        )
        
        assert torch.equal(result_theta, mock_theta)
        assert torch.equal(result_x, mock_x)
    
    def test_abstract_method_enforcement(self):
        """Test that BaseSimulator cannot be instantiated without implementing abstract methods."""
        with pytest.raises(TypeError):
            BaseSimulator("test")
    
    def test_simulate_wrapper_creation(self):
        """Test that the simulator wrapper correctly calls the simulate method."""
        simulator = MockSimulator()
        parameters = torch.tensor([2.0, 3.0])
        
        # Test the wrapper function that would be created in simulate_batch
        def simulator_wrapper(theta):
            return simulator.simulate(theta)
        
        result = simulator_wrapper(parameters)
        expected = parameters ** 2
        assert torch.equal(result, expected)
