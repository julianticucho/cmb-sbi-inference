import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import numpy as np
from sklearn.model_selection import KFold
from .model import CompressionAE  # Asegúrate de que esta importación coincida con tu estructura
from .config import TRAIN_CONFIG, DATA_PATHS
from .utils import prepare_data

def train_model(X_train, X_val, fold):
    # Inicializar modelo
    model = CompressionAE(
        input_size=TRAIN_CONFIG['lmax'] + 1,
        latent_size=TRAIN_CONFIG['latent_dim']
    )
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_CONFIG['learning_rate'])
    criterion = nn.MSELoss()

    # Datasets y DataLoader
    train_dataset = TensorDataset(torch.as_tensor(X_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True)

    # Entrenamiento con early stopping
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(TRAIN_CONFIG['epochs']):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, in train_loader:
            optimizer.zero_grad()
            x_recon, _ = model(batch_X)
            loss = criterion(x_recon, batch_X)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validación
        model.eval()
        with torch.no_grad():
            val_loss = criterion(
                model(torch.as_tensor(X_val, dtype=torch.float32))[0],
                torch.as_tensor(X_val, dtype=torch.float32)
            )

        # Logs por época
        print(f"Fold {fold + 1} | Epoch {epoch + 1}/{TRAIN_CONFIG['epochs']} | "
              f"Train Loss: {epoch_loss/len(train_loader):.6f} | "
              f"Val Loss: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Guardar el mejor modelo de este fold
            model_path = os.path.join(DATA_PATHS['output'], f'best_ae_fold_{fold + 1}.pth')
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= TRAIN_CONFIG['patience']:
                print(f"Early stopping en epoch {epoch + 1} para fold {fold + 1}")
                break

    return best_val_loss

def main():
    # Cargar y preparar datos
    spectra = torch.load(DATA_PATHS['spectra'])
    params = torch.load(DATA_PATHS['params'])
    X, _, _, _, _, _ = prepare_data(spectra, params)  # Solo necesitamos X para el autoencoder

    # Crear splits KFold por cosmologías
    kf = KFold(n_splits=TRAIN_CONFIG['num_cosmologies'])
    unique_cosmologies = np.arange(TRAIN_CONFIG['num_cosmologies'])
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(unique_cosmologies)):
        print(f"\n=== Fold {fold + 1}/{TRAIN_CONFIG['num_cosmologies']} ===")
        
        # Convertir índices de cosmologías a índices de realizaciones
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

        # Entrenar modelo para este fold
        best_loss = train_model(X[train_realizations], X[val_realizations], fold)
        print(f"Mejor pérdida de validación (Fold {fold + 1}): {best_loss:.6f}")

if __name__ == "__main__":
    # Crear directorios de salida si no existen
    os.makedirs(DATA_PATHS['output'], exist_ok=True)
    main()