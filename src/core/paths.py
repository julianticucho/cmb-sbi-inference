from pathlib import Path
from typing import Optional

class Paths:
    _instance = None
    _initialized = False
    
    def __new__(cls, base_dir=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, base_dir: str = None):
        if not self._initialized:
            self.base_dir = Path(base_dir) if base_dir else Path.cwd()
            self._setup_paths()
            self._initialized = True
    
    def _setup_paths(self):
        self.data_dir = self.base_dir / "data"
        self.cobaya_dir = self.data_dir / "cobaya"
        self.planck_dir = self.data_dir / "planck"
        self.planck_data_dir = self.planck_dir 
        self.plots_dir = self.data_dir / "plots"
        self.simulations_dir = self.data_dir / "simulations"
        
        self.results_dir = self.base_dir / "results"
        self.calibration_dir = self.results_dir / "calibration"
        self.chains_dir = self.results_dir / "chains"
        self.confidence_dir = self.results_dir / "confidence"
        self.consistency_dir = self.results_dir / "consistency"
        self.correlation_dir = self.results_dir / "correlation"
        self.last_dir = self.results_dir / "last"
        self.models_dir = self.results_dir / "models"
        self.posteriors_dir = self.results_dir / "posteriors"
        self.summary_dir = self.results_dir / "summary"
        self.synthetic_dir = self.results_dir / "synthetic"
    
    def ensure_directories(self):
        paths = [
            self.data_dir, self.simulations_dir, self.planck_dir,
            self.results_dir, self.models_dir, self.posteriors_dir,
            self.plots_dir, self.correlation_dir, self.calibration_dir,
            self.confidence_dir, self.consistency_dir, self.chains_dir,
            self.synthetic_dir, self.last_dir
        ]
        
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)
    
    def get_path(self, name: str) -> Optional[Path]:
        return getattr(self, f"{name}_dir", None)
    
    def to_dict(self) -> dict:
        return {
            "simulations": str(self.simulations_dir),
            "planck": str(self.planck_dir),
            "models": str(self.models_dir),
            "posteriors": str(self.posteriors_dir),
            "correlation": str(self.correlation_dir),
            "calibration": str(self.calibration_dir),
            "confidence": str(self.confidence_dir),
            "summary": str(self.summary_dir),
            "consistency": str(self.consistency_dir),
            "chains": str(self.chains_dir),
            "last": str(self.last_dir),
            "synthetic": str(self.synthetic_dir),
            "plots": str(self.plots_dir),
            "data": str(self.data_dir),
            "results": str(self.results_dir),
        }
