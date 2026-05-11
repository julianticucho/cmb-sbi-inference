import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, List


def correlation_matrix(
    samples: torch.Tensor,
    param_names: List[str],
    param_labels: Optional[List[str]] = None
) -> plt.Figure:
    samples_np = samples.numpy() if not isinstance(samples, np.ndarray) else samples
    corr_matrix = np.corrcoef(samples_np.T)
    
    if param_labels is None:
        param_labels = [name.replace('_', r'\_') for name in param_names]

    fig = plt.figure(figsize=(6, 4))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        square=True,
        xticklabels=param_labels,
        yticklabels=param_labels,
        vmin=-1, vmax=1
    )
    return fig
