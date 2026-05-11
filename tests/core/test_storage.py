import pytest
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from unittest.mock import Mock, patch
from src.core.paths import Paths
from src.core.storage import StorageManager


class TestStorageManager:
    
    @pytest.fixture
    def storage_manager(self, temp_dir):
        paths = Paths(str(temp_dir))
        paths.ensure_directories()
        return StorageManager(paths)
    
    @pytest.fixture
    def sample_tensors(self):
        theta = torch.randn(10, 5)
        x = torch.randn(10, 3)
        return theta, x
    
    def test_initialization_with_paths(self, temp_dir):
        paths = Paths(str(temp_dir))
        storage = StorageManager(paths)
        assert storage.paths == paths
    
    def test_initialization_without_paths(self):
        """Test StorageManager initialization with default paths."""
        storage = StorageManager()
        assert isinstance(storage.paths, Paths)
    
    def test_save_simulations_success(self, storage_manager, sample_tensors):
        """Test successful simulation saving."""
        theta, x = sample_tensors
        filename = "test_simulations.pt"
        
        storage_manager.save_simulations(theta, x, filename)
        
        # Check file was created
        file_path = storage_manager.paths.simulations_dir / filename
        assert file_path.exists()
        
        # Check file contents
        loaded_data = torch.load(file_path)
        assert torch.equal(loaded_data["theta"], theta)
        assert torch.equal(loaded_data["x"], x)
    
    def test_save_simulations_invalid_filename(self, storage_manager, sample_tensors):
        """Test that save_simulations raises error for invalid filename."""
        theta, x = sample_tensors
        
        with pytest.raises(ValueError, match="Filename must end with .pt"):
            storage_manager.save_simulations(theta, x, "invalid.txt")
    
    def test_load_simulations_success(self, storage_manager, sample_tensors):
        """Test successful simulation loading."""
        theta, x = sample_tensors
        filename = "test_simulations.pt"
        
        # First save the data
        storage_manager.save_simulations(theta, x, filename)
        
        # Then load it
        loaded_theta, loaded_x = storage_manager.load_simulations(filename)
        
        assert torch.equal(loaded_theta, theta)
        assert torch.equal(loaded_x, x)
    
    def test_load_simulations_file_not_found(self, storage_manager):
        """Test that load_simulations raises error for missing file."""
        with pytest.raises(FileNotFoundError, match="File nonexistent.pt not found"):
            storage_manager.load_simulations("nonexistent.pt")
    
    @patch('torch.save')
    def test_save_model(self, mock_torch_save, storage_manager):
        """Test model saving."""
        mock_model = Mock()
        filename = "test_model.pt"
        
        storage_manager.save_model(mock_model, filename)
        
        expected_path = storage_manager.paths.models_dir / filename
        mock_torch_save.assert_called_once_with(mock_model, expected_path)
    
    @patch('torch.load')
    @patch('torch.save')
    def test_load_model_with_state_dict(self, mock_torch_save, mock_torch_load, storage_manager):
        """Test model loading with state dictionary."""
        mock_model = Mock()
        mock_state_dict = {"param": torch.randn(5)}
        mock_torch_load.return_value = mock_state_dict
        
        theta = torch.randn(10, 3)
        x = torch.randn(10, 2)
        filename = "test_model.pt"
        
        result = storage_manager.load_model(
            filename, 
            model=mock_model, 
            theta=theta, 
            x=x, 
            load_state_dict=True
        )
        
        # Verify method calls
        mock_model.append_simulations.assert_called_once_with(theta, x)
        mock_model.train.assert_called_once_with(max_num_epochs=0)
        mock_model.load_state_dict.assert_called_once_with(mock_state_dict)
        mock_model.build_posterior.assert_called_once()
    
    @patch('torch.load')
    def test_load_model_without_state_dict(self, mock_torch_load, storage_manager):
        """Test model loading without state dictionary."""
        mock_model = Mock()
        mock_model.build_posterior.return_value = "built_model"
        mock_torch_load.return_value = mock_model
        
        filename = "test_model.pt"
        result = storage_manager.load_model(filename)
        
        mock_torch_load.assert_called_once_with(storage_manager.paths.models_dir / filename)
        mock_model.build_posterior.assert_called_once()
        assert result == "built_model"
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_save_ppc(self, mock_close, mock_savefig, storage_manager):
        """Test saving posterior predictive check plots."""
        mock_fig = Mock()
        filename = "test_ppc.png"
        
        storage_manager.save_ppc(mock_fig, filename)
        
        expected_path = storage_manager.paths.confidence_dir / filename
        mock_savefig.assert_called_once_with(expected_path, bbox_inches='tight')
        mock_close.assert_called_once_with(mock_fig)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_save_diagnostic(self, mock_close, mock_savefig, storage_manager):
        """Test saving diagnostic plots."""
        mock_fig = Mock()
        filename = "test_diagnostic.png"
        
        storage_manager.save_diagnostic(mock_fig, filename)
        
        expected_path = storage_manager.paths.calibration_dir / filename
        mock_savefig.assert_called_once_with(expected_path, bbox_inches='tight')
        mock_close.assert_called_once_with(mock_fig)
