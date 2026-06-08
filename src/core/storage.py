import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

class StorageManager:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, base_dir: Optional[str] = None):
        if not self._initialized:
            self.base_dir = Path(base_dir) if base_dir else Path.cwd()
            self._setup_paths()
            self._initialized = True

    def _setup_paths(self):
        self.data_dir = self.base_dir / "data"
        self.results_dir = self.base_dir / "results"

        self.dirs = {
            "simulations": self.data_dir / "simulations",
            "pipelines": self.data_dir / "pipelines", # For processed data
            "cobaya": self.data_dir / "cobaya",
            "planck": self.data_dir / "planck",
            "plots": self.data_dir / "plots",
            "models": self.results_dir / "models",
            "sequential_models": self.results_dir / "sequential_models",
            "models_per_round": self.results_dir / "models_per_round",
            "posteriors": self.results_dir / "posteriors",
            "diagnostics": self.results_dir / "diagnostics",
            "hpd": self.results_dir / "hpd",
            "data_ppc": self.results_dir / "data_ppc",
            "chains": self.results_dir / "chains",
            "calibration": self.results_dir / "calibration",
            "confidence": self.results_dir / "confidence",
            "consistency": self.results_dir / "consistency",
            "correlation": self.results_dir / "correlation",
            "synthetic": self.results_dir / "synthetic",
            "last": self.results_dir / "last",
            "summary": self.results_dir / "summary",
            "embedding_nn": self.results_dir / "embedding_nn",
            "regression": self.results_dir / "regression",
        }

    def ensure_directories(self):
        for path in self.dirs.values():
            path.mkdir(parents=True, exist_ok=True)

    def get_dir(self, category: str) -> Path:
        if category not in self.dirs:
            raise ValueError(f"Unknown category: {category}. Available: {list(self.dirs.keys())}")
        return self.dirs[category]

    def _get_path(self, category: str, filename: str) -> Path:
        return self.get_dir(category) / filename

    # --- Persistence Methods ---

    def save_simulations(self, theta: torch.Tensor, x: torch.Tensor, filename: str):
        if not filename.endswith('.pt'):
            raise ValueError("Filename must end with .pt")
        path = self._get_path("simulations", filename)
        torch.save({"theta": theta, "x": x}, path)
    
    def load_simulations(self, filename: str) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self._get_path("simulations", filename)
        if not path.exists():
            # Try in pipelines dir if not in simulations (for processed data)
            path = self._get_path("pipelines", filename)
            if not path.exists():
                raise FileNotFoundError(f"File {filename} not found in simulations or pipelines")
        
        data = torch.load(path, weights_only=True)
        return data["theta"], data["x"]

    def load_multiple_simulations(self, filenames: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        theta_list, x_list = [], []
        for filename in filenames:
            theta, x = self.load_simulations(filename)
            theta_list.append(theta)
            x_list.append(x)
        return torch.cat(theta_list, dim=0), torch.cat(x_list, dim=0)
        
    def save_model(
        self, 
        model: torch.nn.Module,
        simulation_files: List[str],
        prior_type: str,
        inference_type: str,
        filename: str,
        embedding_nn_filename: Optional[str] = None,
    ):
        path = self._get_path("models", filename)
        cfg = {
            "state_dict": model.state_dict(),
            "simulation_files": simulation_files,
            "prior_type": prior_type,
            "inference_type": inference_type,
            "embedding_nn_filename": embedding_nn_filename,
        }
        torch.save(cfg, path)

    def save_sequential_model(
        self,
        state_dicts: List[Dict[str, Any]],
        simulation_files: List[str],
        prior_type: str,
        inference_type: str,
        simulator_type: str,
        pipeline_type: str,
        x_obs: torch.Tensor,
        filename: str,
    ):
        """
        Saves a master file containing the history of a sequential training session.
        References external simulation files.
        """
        path = self._get_path("sequential_models", filename)
        cfg = {
            "state_dicts": state_dicts,
            "simulation_files": simulation_files,
            "prior_type": prior_type,
            "inference_type": inference_type,
            "simulator_type": simulator_type,
            "pipeline_type": pipeline_type,
            "x_obs": x_obs,
            "num_rounds": len(state_dicts),
        }
        torch.save(cfg, path)

    def save_model_per_round(
        self,
        model: torch.nn.Module,
        round: int,
        simulation_files: List[str],
        prior_type: str,
        inference_type: str,
        filename: str,
        x_obs: Optional[torch.Tensor] = None,
        previous_filename: Optional[str] = None,
    ):
        path = self._get_path("models_per_round", filename)
        cfg = {
            "state_dict": model.state_dict(),
            "round": round,
            "simulation_files": simulation_files,
            "prior_type": prior_type,
            "inference_type": inference_type,
            "x_obs": x_obs,
            "previous_filename": previous_filename,
        }
        torch.save(cfg, path)

    def load_model(self, filename: str) -> Dict[str, Any]:
        path = self._get_path("models", filename)
        if not path.exists():
            raise FileNotFoundError(f"Model file {filename} not found")
        return torch.load(path, weights_only=True)

    def load_sequential_model(self, filename: str) -> Dict[str, Any]:
        path = self._get_path("sequential_models", filename)
        if not path.exists():
            # Try in models dir for compatibility
            path = self._get_path("models", filename)
            
        if not path.exists():
            raise FileNotFoundError(f"Sequential model file {filename} not found")
        return torch.load(path, weights_only=False)
    
    def load_model_per_round(self, filename: str) -> Dict[str, Any]:
        path = self._get_path("models_per_round", filename)
        if not path.exists():
            raise FileNotFoundError(f"Model file {filename} not found")
        return torch.load(path, weights_only=False)
    
    def save_figure(self, fig: plt.Figure, filename: str, category: str = "plots"):
        path = self._get_path(category, filename)
        plt.savefig(path, bbox_inches='tight')

    def save_embedding_nn(self, 
        nn: torch.nn.Module,
        history: Dict[str, Any],
        simulation_files: List[str],
        dataloader_name: str,
        model_name: str,
        filename: str,
    ):
        path = self._get_path("embedding_nn", filename)
        cfg = {
            "state_dict": nn.state_dict(),
            "history": history,
            "simulation_files": simulation_files,
            "dataloader_name": dataloader_name,
            "model_name": model_name,
        }
        torch.save(cfg, path)

    def load_embedding_nn(self, filename: str) -> Dict[str, Any]:
        """Load a checkpoint saved by ``save_embedding_nn``.

        Returns a dict with keys: ``state_dict``, ``history``,
        ``simulation_files``, ``dataloader_name``, ``model_name``.
        """
        path = self._get_path("embedding_nn", filename)
        if not path.exists():
            raise FileNotFoundError(f"Embedding NN file '{filename}' not found in {path.parent}")
        return torch.load(path, weights_only=False)

storage = StorageManager()
