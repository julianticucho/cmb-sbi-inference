import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import numpy as np
from sklearn.model_selection import KFold
from .model import Compressor  
from .config import TRAIN_CONFIG, DATA_PATHS
from .utils import prepare_data

def train_model(X_train, theta_train, X_val, theta_val, fold):
    model = Compressor(
        input_size=TRAIN_CONFIG['lmax'] + 1,
        latent_size=TRAIN_CONFIG['latent_dim'],
        param_dim=theta_train.shape[1]
    )
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_CONFIG['learning_rate'])
    train_dataset = TensorDataset(torch.as_tensor(X_train, dtype=torch.float32), torch.as_tensor(theta_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(TRAIN_CONFIG['epochs']):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_theta in train_loader:
            optimizer.zero_grad()
            compressed = model(batch_X)
            loss = model.compute_loss(compressed, batch_theta) # Función de pérdida del paper de Jeffrey
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_compressed = model(torch.as_tensor(X_val, dtype=torch.float32))
            val_loss = model.compute_loss(val_compressed, torch.as_tensor(theta_val, dtype=torch.float32))

        print(f"Fold {fold + 1} | Epoch {epoch + 1}/{TRAIN_CONFIG['epochs']} | "
              f"Train Loss: {epoch_loss/len(train_loader):.6f} | "
              f"Val Loss: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model_path = os.path.join(DATA_PATHS['output'], f'best_ae_fold_{fold + 1}.pth')
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= TRAIN_CONFIG['patience']:
                print(f"Early stopping en epoch {epoch + 1} para fold {fold + 1}")
                break

    return best_val_loss

def main():
    spectra = torch.load(DATA_PATHS['spectra'])
    params = torch.load(DATA_PATHS['params'])
    X, theta, _, _, _, _ = prepare_data(spectra, params)  

    # Splits KFold por cosmologías
    kf = KFold(n_splits=TRAIN_CONFIG['num_cosmologies'])
    unique_cosmologies = np.arange(TRAIN_CONFIG['num_cosmologies'])
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(unique_cosmologies)):
        print(f"\n=== Fold {fold + 1}/{TRAIN_CONFIG['num_cosmologies']} ===")
        
        train_realizations = []
        for cosmo_idx in train_idx:
            start = cosmo_idx * TRAIN_CONFIG['realizations_per_cosmology']
            end = start + TRAIN_CONFIG['realizations_per_cosmology']
            train_realizations.extend(range(start, end))
        
        val_realizations = []
        for cosmo_idx in val_idx:
            start = cosmo_idx * TRAIN_CONFIG['realizations_per_cosmology']
            end = start + TRAIN_CONFIG['realizations_per_cosmology']
            val_realizations.extend(range(start, end))

        best_loss = train_model(X[train_realizations], theta[train_realizations], X[val_realizations], theta[val_realizations], fold)
        print(f"Mejor pérdida de validación (Fold {fold + 1}): {best_loss:.6f}")

if __name__ == "__main__":
    os.makedirs(DATA_PATHS['output'], exist_ok=True)
    main()