from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader


class BaseModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def train_model(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        max_epochs: int = 2000,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        patience: int = 20,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        clip_max_norm: Optional[float] = 5.0,
        device: Optional[str] = None,
    ) -> dict:
        """Train the model with early stopping based on validation loss.

        Args:
            train_dataloader: DataLoader for training data.
            val_dataloader: DataLoader for validation data.
            max_epochs: Hard upper limit on the number of training epochs.
            lr: Learning rate for the Adam optimizer.
            weight_decay: L2 regularization coefficient for Adam.
            patience: Number of epochs without improvement in validation loss
                before stopping early.
            min_delta: Minimum absolute decrease in validation loss to be
                considered an improvement. Helps ignore negligible changes.
            restore_best_weights: If True, restores the model weights from the
                epoch with the lowest validation loss after training ends.
            clip_max_norm: Maximum norm for gradient clipping. Use None to
                disable clipping.
            device: Device to train on. Defaults to CUDA if available.

        Returns:
            dict with keys:
                - "train_loss": list of per-epoch training losses.
                - "val_loss": list of per-epoch validation losses.
                - "best_epoch": epoch index (1-based) with the best val loss.
                - "stopped_early": True if training stopped before max_epochs.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Training on {device}")
        self.to(device)

        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        criterion = nn.MSELoss()

        history = {"train_loss": [], "val_loss": [], "best_epoch": 1, "stopped_early": False}

        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        epoch = 0

        while epoch < max_epochs:
            epoch += 1

            # training
            self.train()
            train_loss = 0.0
            for x_batch, theta_batch in train_dataloader:
                x_batch = x_batch.to(device)
                theta_batch = theta_batch.to(device)

                optimizer.zero_grad()
                pred = self(x_batch)
                loss = criterion(pred, theta_batch)
                loss.backward()

                if clip_max_norm is not None:
                    clip_grad_norm_(self.parameters(), max_norm=clip_max_norm)

                optimizer.step()
                train_loss += loss.item() * x_batch.size(0)

            train_loss /= len(train_dataloader.dataset)  

            # validation
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, theta_batch in val_dataloader:
                    x_batch = x_batch.to(device)
                    theta_batch = theta_batch.to(device)
                    pred = self(x_batch)
                    loss = criterion(pred, theta_batch)
                    val_loss += loss.item() * x_batch.size(0)

            val_loss /= len(val_dataloader.dataset)  

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            # early stopping
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_state = deepcopy(self.state_dict())
                history["best_epoch"] = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # logging
            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch [{epoch:>{len(str(max_epochs))}}/{max_epochs}] "
                    f"train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f} | "
                    f"no improvement: {epochs_without_improvement}/{patience}"
                )

            # check convergence
            if epochs_without_improvement >= patience:
                history["stopped_early"] = True
                print(
                    f"Early stopping at epoch {epoch} — "
                    f"no val_loss improvement for {patience} epochs "
                    f"(best val_loss: {best_val_loss:.6f} at epoch {history['best_epoch']})"
                )
                break

        # restore best weights
        if restore_best_weights and best_state is not None:
            self.load_state_dict(best_state)

        return history

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: Optional[str] = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        self.to(device)
