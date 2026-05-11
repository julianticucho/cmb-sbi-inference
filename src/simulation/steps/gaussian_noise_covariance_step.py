import torch
from typing import Optional
from ..contracts.base_step import BaseStep


class GaussianNoiseCovarianceStep(BaseStep):
    
    def __init__(self, cov_matrix: torch.Tensor):
        super().__init__("NoiseCovarianceStep")
        self.cov_matrix = cov_matrix
        self._validate_covariance()
        self.cholesky_L = torch.linalg.cholesky(cov_matrix)
    
    def _validate_covariance(self):
        if self.cov_matrix.ndim != 2:
            raise ValueError("Covariance matrix must be 2D")
        if self.cov_matrix.shape[0] != self.cov_matrix.shape[1]:
            raise ValueError("Covariance matrix must be square")
        
        try:
            torch.linalg.cholesky(self.cov_matrix)
        except torch.linalg.LinAlgError:
            raise ValueError("Covariance matrix must be positive definite")
    
    def apply(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.ndim != 1:
            raise ValueError("Input must be 1D tensor")
        if x.shape[0] != self.cov_matrix.shape[0]:
            raise ValueError(f"Input length {x.shape[0]} must match covariance matrix size {self.cov_matrix.shape[0]}")
        
        z = torch.randn(x.shape[0], device=x.device)
        noise = self.cholesky_L @ z
        
        return x + noise
