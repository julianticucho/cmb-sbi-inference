from typing import Callable, Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


class DataLoaderFactory:
    def __init__(self, theta: torch.Tensor, x: torch.Tensor):
        self._theta = theta
        self._x = x

    def get_available_configurations(self) -> Dict[str, Callable]:
        return {
            "raw": self.create_raw,
            "flat": self.create_flat,
            "normalized": self.create_normalized,
            "flat_normalized": self.create_flat_normalized,
        }

    def get_dataloader(
        self,
        config: str,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        configs = self.get_available_configurations()
        if config not in configs:
            raise ValueError(
                f"Config '{config}' not found. "    
                f"Available: {list(configs.keys())}"
            )
        return configs[config]()

    def create_raw(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        return self._make_loaders(
            self._x, batch_size=256, val_split=0.1, 
            test_split=0.1, shuffle=True, seed=None
        )

    def create_flat(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        x_flat = self._x.view(self._x.size(0), -1).float()
        return self._make_loaders(
            x_flat, batch_size=256, val_split=0.1, 
            test_split=0.1, shuffle=True, seed=None
        )

    def create_normalized(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        x_flat = self._x.view(self._x.size(0), -1).float()
        mean = x_flat.mean(dim=0, keepdim=True)
        std = x_flat.std(dim=0, keepdim=True).clamp(min=1e-8)
        x_norm = ((x_flat - mean) / std).view(self._x.shape)
        return self._make_loaders(
            x_norm, batch_size=256, val_split=0.1, 
            test_split=0.1, shuffle=True, seed=None
        )

    def create_flat_normalized(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        x_flat = self._x.view(self._x.size(0), -1).float()
        mean = x_flat.mean(dim=0, keepdim=True)
        std = x_flat.std(dim=0, keepdim=True).clamp(min=1e-8)
        x_norm = (x_flat - mean) / std
        return self._make_loaders(
            x_norm, batch_size=256, val_split=0.1, 
            test_split=0.1, shuffle=True, seed=None
        )

    def _make_loaders(
        self,
        x: torch.Tensor,
        batch_size: int,
        val_split: float,
        test_split: float,
        shuffle: bool,
        seed: Optional[int],
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        dataset = TensorDataset(x, self._theta)
        train_ds, val_ds, test_ds = self._split_dataset(
            dataset, val_split, test_split, seed
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def _split_dataset(
        self,
        dataset: TensorDataset,
        val_split: float,
        test_split: float,  
        seed: Optional[int],
    ):
        n = len(dataset)
        n_test = int(n * test_split)
        n_val = int(n * val_split)
        n_train = n - n_val - n_test
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        return random_split(dataset, [n_train, n_val, n_test], generator=generator)

