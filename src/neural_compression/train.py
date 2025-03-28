import optuna
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from .model import CosmologicalNetwork
from .config import TRAIN_CONFIG, DATA_PATHS, PARAM_NAMES
from .utils import prepare_data, create_kfold_split
import os
import numpy as np

def objective(trial, X_train, y_train, X_val, y_val):
    # Hiperparámetros a optimizar
    hidden_layers = trial.suggest_int('hidden_layers', 1, 5)
    hidden_units = trial.suggest_int('hidden_units', 32, 512)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    
    model = CosmologicalNetwork(
        input_size=X_train.shape[1],
        output_size=y_train.shape[1],
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
        dropout_rate=dropout_rate
    )
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Entrenamiento con early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(TRAIN_CONFIG['epochs']):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validación
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(torch.FloatTensor(X_val)), torch.FloatTensor(y_val))
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= TRAIN_CONFIG['patience']:
                break
    
    return best_val_loss.item()

def main():
    # Cargar datos con seguridad
    spectra = torch.load(DATA_PATHS['spectra'], weights_only=True)
    params = torch.load(DATA_PATHS['params'], weights_only=True)
    
    # Verificación de dimensiones
    assert spectra.shape == (5000, 2401), f"Forma inesperada de espectros: {spectra.shape}"
    assert params.shape == (5000, 6), f"Forma inesperada de parámetros: {params.shape}"
    
    # Preparar datos
    X, y, s_mean, s_std, p_mean, p_std = prepare_data(spectra, params)
    
    # Crear splits por cosmología
    kf = create_kfold_split(X, TRAIN_CONFIG['num_cosmologies'], TRAIN_CONFIG['realizations_per_cosmology'])
    
    for fold, (train_groups_idx, test_group_idx) in enumerate(kf):
        # Convertir índices de grupos a índices de realizaciones
        train_idx = []
        for group_idx in train_groups_idx:
            start = group_idx * TRAIN_CONFIG['realizations_per_cosmology']
            end = start + TRAIN_CONFIG['realizations_per_cosmology']
            train_idx.extend(range(start, end))
        
        test_idx = []
        for group_idx in test_group_idx:
            start = group_idx * TRAIN_CONFIG['realizations_per_cosmology']
            end = start + TRAIN_CONFIG['realizations_per_cosmology']
            test_idx.extend(range(start, end))
        
        print(f"\nFold {fold+1}:")
        print(f" - Cosmologías entrenamiento: {train_groups_idx}")
        print(f" - Cosmología prueba: {test_group_idx}")
        
        # Dividir train en train/val (80/20 de las realizaciones de entrenamiento)
        val_size = int(0.2 * len(train_idx))
        np.random.shuffle(train_idx)  # Mezclar antes de dividir
        train_idx, val_idx = train_idx[:-val_size], train_idx[-val_size:]
        
        # Optimización con Optuna
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: objective(
                trial, 
                X[train_idx], 
                y[train_idx],
                X[val_idx],
                y[val_idx]
            ),
            n_trials=TRAIN_CONFIG['optuna_trials']
        )
        
        # Guardar mejor modelo
        best_params = study.best_params
        model = CosmologicalNetwork(
            input_size=X.shape[1],
            output_size=y.shape[1],
            hidden_layers=best_params['hidden_layers'],
            hidden_units=best_params['hidden_units'],
            dropout_rate=best_params['dropout_rate']
        )
        
        # Guardar modelo y estadísticas
        os.makedirs(DATA_PATHS['output'], exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'best_params': best_params,
            'spectra_mean': s_mean,
            'spectra_std': s_std,
            'params_mean': p_mean,
            'params_std': p_std
        }, f"{DATA_PATHS['output']}model_fold_{fold}.pth")

if __name__ == "__main__":
    main()