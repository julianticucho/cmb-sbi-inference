import torch
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from .paths import Paths 

class StorageManager:

    def __init__(self, paths: Optional[Paths] = None):
        self.paths = paths or Paths()
    
    def save_simulations(self, theta: torch.Tensor, x: torch.Tensor, filename: str):
        if not filename.endswith('.pt'):
            raise ValueError("Filename must end with .pt")
        path = self.paths.simulations_dir / filename
        torch.save({"theta": theta, "x": x}, path)
    
    def load_simulations(self, filename: str) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.paths.simulations_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"File {filename} not found")
        data = torch.load(path, weights_only=True)
        theta, x = data["theta"], data["x"]
        return theta, x

    def load_multiple_simulations(self, filenames: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        theta_list, x_list = [], []
        for filename in filenames:
            path = self.paths.simulations_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"File {filename} not found")
            data = torch.load(path, weights_only=True)
            theta_list.append(data["theta"])
            x_list.append(data["x"])
        theta = torch.cat(theta_list, dim=0)
        x = torch.cat(x_list, dim=0)
        return theta, x
        
    def save_model(
        self, 
        model: torch.nn.Module,
        simulation_files: List[str],
        prior_type: str,
        inference_type: str,
        filename: str, 
    ):
        path = self.paths.models_dir / filename
        cfg = {
            "state_dict": model.state_dict(),
            "simulation_files": simulation_files,
            "prior_type": prior_type,
            "inference_type": inference_type,
        }
        torch.save(cfg, path)

    def load_model(
        self, 
        filename: str, 
    ) -> torch.nn.Module:
        path = self.paths.models_dir / filename
        cfg = torch.load(path, weights_only=True)
        state_dict = cfg["state_dict"]
        simulation_files = cfg["simulation_files"]
        prior_type = cfg["prior_type"]
        inference_type = cfg["inference_type"]
        
        return state_dict, simulation_files, prior_type, inference_type

    def save_ppc(self, fig: plt.Figure, filename: str):
        path = self.paths.confidence_dir / filename
        plt.savefig(path, bbox_inches='tight')

    def save_diagnostic(self, fig: plt.Figure, filename: str):
        path = self.paths.calibration_dir / filename
        plt.savefig(path, bbox_inches='tight')
