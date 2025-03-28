import torch
import numpy as np
from .model import CosmologicalNetwork
from .config import DATA_PATHS, PARAM_NAMES
from .utils import prepare_data
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import os

def load_models(num_folds=10):
    """Carga todos los modelos entrenados"""
    models = []
    stats = []
    
    for fold in range(num_folds):
        checkpoint = torch.load(f"{DATA_PATHS['output']}model_fold_{fold}.pth")
        
        model = CosmologicalNetwork(
            input_size=2401,  
            output_size=6,    
            hidden_layers=checkpoint['best_params']['hidden_layers'],
            hidden_units=checkpoint['best_params']['hidden_units'],
            dropout_rate=checkpoint['best_params']['dropout_rate']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        models.append(model)
        stats.append({
            'spectra_mean': checkpoint['spectra_mean'],
            'spectra_std': checkpoint['spectra_std'],
            'params_mean': checkpoint['params_mean'],
            'params_std': checkpoint['params_std']
        })
    
    return models, stats

def denormalize_params(y_norm, params_mean, params_std):
    """Revertir la normalización de los parámetros"""
    return y_norm * params_std + params_mean

def test_models():
    # Cargar datos
    spectra = torch.load(DATA_PATHS['spectra'], weights_only=True)
    params = torch.load(DATA_PATHS['params'], weights_only=True)
    
    # Cargar modelos
    models, stats = load_models()
    
    # Preparar datos (sin normalizar, ya que los modelos manejan esto internamente)
    X = spectra.numpy()
    y_true = params.numpy()
    
    # Realizar predicciones con cada modelo
    all_predictions = []
    
    for model, stat in zip(models, stats):
        with torch.no_grad():
            # Normalizar entrada
            X_norm = (X - stat['spectra_mean']) / (stat['spectra_std'] + 1e-8)
            y_pred_norm = model(torch.FloatTensor(X_norm)).numpy()
            # Desnormalizar predicciones
            y_pred = denormalize_params(y_pred_norm, stat['params_mean'], stat['params_std'])
            all_predictions.append(y_pred)
    
    # Calcular promedio de predicciones de todos los modelos
    y_pred_avg = np.mean(all_predictions, axis=0)
    
    # Calcular métricas
    mse = mean_squared_error(y_true, y_pred_avg)
    r2 = r2_score(y_true, y_pred_avg)
    
    print(f"\nMétricas de evaluación:")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Visualización por parámetro
    plt.figure(figsize=(15, 10))
    for i, param_name in enumerate(PARAM_NAMES):
        plt.subplot(2, 3, i+1)
        plt.scatter(y_true[:, i], y_pred_avg[:, i], alpha=0.5)
        plt.plot([y_true[:, i].min(), y_true[:, i].max()], 
                 [y_true[:, i].min(), y_true[:, i].max()], 'r--')
        plt.xlabel(f'True {param_name}')
        plt.ylabel(f'Predicted {param_name}')
        plt.title(f'{param_name} (R²={r2_score(y_true[:, i], y_pred_avg[:, i]):.3f})')
    plt.tight_layout()
    
    # Guardar figuras
    os.makedirs("results/plots/", exist_ok=True)
    plt.savefig("results/plots/predictions_vs_true.png")
    plt.show()
    
    return mse, r2

if __name__ == "__main__":
    test_models()