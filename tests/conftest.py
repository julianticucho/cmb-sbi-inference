import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    return {
        'parameters': np.random.randn(100, 5),
        'simulations': np.random.randn(100, 10),
        'observations': np.random.randn(10)
    }


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        'n_samples': 100,
        'n_dimensions': 5,
        'seed': 42,
        'output_dir': '/tmp/test_output'
    }
