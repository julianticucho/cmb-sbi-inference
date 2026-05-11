import pytest
from pathlib import Path
from src.core.paths import Paths


@pytest.fixture(autouse=True)
def reset_paths_singleton():
    Paths._instance = None
    Paths._initialized = False
    yield


class TestPaths:
    
    def test_singleton_pattern(self):
        paths1 = Paths()
        paths2 = Paths()
        assert paths1 is paths2
    
    def test_initialization_with_base_dir(self, temp_dir):
        paths = Paths(str(temp_dir))
        assert paths.base_dir == temp_dir

    def test_initialization_without_base_dir(self):
        paths = Paths()
        assert paths.base_dir == Path.cwd()
    
    def test_setup_paths(self, temp_dir):
        paths = Paths(str(temp_dir))
        
        assert paths.data_dir == temp_dir / "data"
        assert paths.cobaya_dir == temp_dir / "data" / "cobaya"
        assert paths.planck_dir == temp_dir / "data" / "planck"
        assert paths.plots_dir == temp_dir / "data" / "plots"
        assert paths.simulations_dir == temp_dir / "data" / "simulations"
        
        assert paths.results_dir == temp_dir / "results"
        assert paths.models_dir == temp_dir / "results" / "models"
        assert paths.posteriors_dir == temp_dir / "results" / "posteriors"
        assert paths.calibration_dir == temp_dir / "results" / "calibration"
    
    def test_ensure_directories(self, temp_dir):
        paths = Paths(str(temp_dir))
        paths.ensure_directories()
        
        assert paths.data_dir.exists()
        assert paths.results_dir.exists()
        assert paths.simulations_dir.exists()
        assert paths.models_dir.exists()
        assert paths.posteriors_dir.exists()
    
    def test_get_path(self, temp_dir):
        paths = Paths(str(temp_dir))
        
        assert paths.get_path("data") == paths.data_dir
        assert paths.get_path("results") == paths.results_dir
        assert paths.get_path("simulations") == paths.simulations_dir
        assert paths.get_path("nonexistent") is None
    
    def test_to_dict(self, temp_dir):
        paths = Paths(str(temp_dir))
        paths_dict = paths.to_dict()
        
        assert isinstance(paths_dict, dict)
        assert "simulations" in paths_dict
        assert "results" in paths_dict
        assert "data" in paths_dict
        assert paths_dict["simulations"] == str(paths.simulations_dir)
        assert paths_dict["results"] == str(paths.results_dir)
