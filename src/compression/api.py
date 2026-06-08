import torch
from typing import Tuple, Dict, List, Optional
from src.compression.factories.dataloader_factory import DataLoaderFactory
from src.compression.factories.model_factory import ModelFactory
from ..core import storage

def train_embedding_nn(
    input_files: List[str],
    dataloader_name: str,
    model_name: str,
    max_epochs: int = 2000,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    patience: int = 20,
    min_delta: float = 0.0,
    restore_best_weights: bool = True,
    clip_max_norm: Optional[float] = 5.0,
    output_name: Optional[str] = None,
    ):
    theta, x = storage.load_multiple_simulations(input_files)
    dataloader_factory = DataLoaderFactory(theta=theta, x=x)
    train_loader, val_loader, _ = dataloader_factory.get_dataloader(dataloader_name)
    model = ModelFactory.get_model(model_name)
    history = model.train_model(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        max_epochs=max_epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=restore_best_weights,
        clip_max_norm=clip_max_norm,
    )
    if output_name is not None:
        storage.save_embedding_nn(
            nn=model,
            history=history,
            simulation_files=input_files,
            dataloader_name=dataloader_name,
            model_name=model_name,
            filename=output_name,
        )
    return history