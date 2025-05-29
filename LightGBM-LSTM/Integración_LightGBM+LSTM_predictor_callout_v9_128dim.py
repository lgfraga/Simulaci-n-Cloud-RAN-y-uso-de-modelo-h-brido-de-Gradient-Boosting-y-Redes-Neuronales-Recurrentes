#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Autor: Luis González Fraga
Ejemplo de integración de LightGBM (como herramienta de ingeniería de características) + LSTM (en PyTorch) para predecir el tráfico de enlace de subida (callout).
Descripción de las secciones del script:
1. Carga el dataset 'dataset_callout.csv', ordena cronológicamente por fecha, hora, minuto y CellID. Normaliza los datos de 'callout' con MinMaxScaler y genera características cíclicas (hour_sin, hour_cos, weekday_sin, weekday_cos) 
   y lags (retardos desde lag1 a lag4) a partir de 'callout_scaled'.
2. Importa un modelo LightGBM preentrenado ('LightGBM_callout_XXX.pkl'), selecciona las 8 características más relevantes y las normaliza con otro MinMaxScaler. Exporta ambos escaladores para ser utilizados en el servicio predictor.
3. Se crean secuencias temporales de 10 minutos (valor óptimo: seq_len=24) usando las características normalizadas y 'callout_scaled' como objetivo (target).
4. Divide los datos en entrenamiento (60%), validación (20%) y prueba (20%), preservando siempre el orden temporal.
5. Define un modelo LSTM en PyTorch con 3 capas, 128 dimensiones ocultas y un dropout de 0.1. Se define y configura el hardware GPU/CPU.
6. Se entrena el LSTM con un DataLoader (valor óptimo: batch_size=32), usa un early stopping basado en la pérdida de validación y luego grafica la evolución del MSE para los datos en la escala normalizada.
7. Evalúa el modelo en entrenamiento, validación y prueba con métricas MSE, MAE, RMSE, R^2 (MSE es la métrica más importante de esta etapa) en escala normalizada y mide también el tiempo de inferencia.
8. Genera dos subfiguras: una con 2016 intervalos (correspondiente a 14 días) mostrando predicciones de entrenamiento, validación y prueba, otra con zoom en índices 1728-1872 (correspondiente a 1 día) para prueba. 
   Calcula SMAPE en escala original que será la métrica principal para evaluar la precisión en la predicción en la escala normal.
Asunciones:
- Requiere Python >= 3.13, PyTorch con GPU, y archivos 'LightGBM_callout_XXX.pkl' y 'dataset_callout.csv' en la misma raíz del directorio del script.
"""

import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from joblib import dump

# -------------------------------------------------------------------------
# 1. CARGAR EL DATASET
# -------------------------------------------------------------------------
df = pd.read_csv("dataset_callout.csv")
print(f"Total de filas en dataset_callout.csv: {len(df)}")

# 1.1 Ordenar el dataset temporalmente
df.sort_values(by=['date', 'hour', 'minute', 'CellID'], inplace=True)
df.reset_index(drop=True, inplace=True)

# 1.2 Normalizar el objetivo (target) 'callout' con la función MinMaxScaler
target_scaler = MinMaxScaler()
df['callout_scaled'] = target_scaler.fit_transform(df[['callout']])

# 1.3 Crear las columnas para la ingeniería de características (las mismas que usadas en LightGBM + lag3 + lag4)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
# Usar 'callout_scaled' para generar los lags, como bien se hizo durante el entrenamiento de LightGBM
df['lag1'] = df['callout_scaled'].shift(1)
df['lag2'] = df['callout_scaled'].shift(2)
df['lag3'] = df['callout_scaled'].shift(3)  # Se ha añadido lag3
df['lag4'] = df['callout_scaled'].shift(4)  # Se ha añadido lag4
df.dropna(inplace=True)

# -------------------------------------------------------------------------
# 2. IMPORTAR LightGBM_callout.pkl
# -------------------------------------------------------------------------
print("\nCargando modelo LightGBM (ingeniería de características)...")
lightgbm_model = joblib.load("LightGBM_callout_001.pkl")

# 2.1 LightGBM: Definiendo las 8 características más relevantes
important_feats = ['hour_sin', 'hour_cos', 'lag1', 'lag2', 'weekday_sin', 'weekday_cos', 'lag3', 'lag4']
print("Características más importantes LightGBM:", important_feats)

# 2.2 Filtrar el dataframe para dejar solo las columnas de interés
df = df[['date', 'hour', 'minute', 'CellID', 'weekday', 'idx', 'callout', 'callout_scaled'] + important_feats].copy()
df.dropna(inplace=True)

# 2.3 Normalizar las características relevantes usando un nuevo MinMaxScaler
scaler = MinMaxScaler()
df[important_feats] = scaler.fit_transform(df[important_feats])

# 2.4 Guardar los escaladores para usarlos posteriormente en el servicio predictor LSTM
dump(scaler, "scaler_callout_features_001.pkl")
dump(target_scaler, "scaler_callout_target_001.pkl")
print("\nNormalización de las características de entrenamiento. Exportado como: scaler_callout_features_001.pkl")
print("Normalización del target 'callout'. Exportado como: scaler_callout_target_001.pkl")

# -------------------------------------------------------------------------
# 3. PREPARAR DATOS PARA LSTM (utilizando secuencias temporales de 10 min)
# -------------------------------------------------------------------------
# 3.1 Se establece que cada fila es un instante (10 min). Se crea una secuencia de longitud seq_len.
seq_len = 24  # Rango de secuencias sugeridas: 4 a 24 (a mayor seq_len mejor precisión a costa de un mayor coste computacional)

# 3.2 Convertir las características más importantes a un array y obtener el objetivo de interés (objetivo: callout_scaled)
data_array = df[important_feats].values
target_array = df['callout_scaled'].values  # Usar callout_scaled como objetivo en la predicción

def create_sequences(features, target, seq_length):
    Xs, ys = [], []
    for i in range(len(features) - seq_length):
        Xs.append(features[i:(i + seq_length), :])
        ys.append(target[i + seq_length])  # Aquí se predice el valor posterior (*IMPORTANTE)
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(data_array, target_array, seq_len)
print("\n--- SECUENCIAS (X = No. de filas, No. de secuencia de 10 min, No. de características), (Y = No. de filas) ---")
print(f"\nSecuencias generadas: X_seq.shape={X_seq.shape}, y_seq.shape={y_seq.shape}")

# -------------------------------------------------------------------------
# 4. DIVISIÓN DEL DATASET EN: ENTRENAMIENTO, VALIDACIÓN Y PRUEBA (TRAIN/VAL/TEST -> 60%-20%-20%)
# -------------------------------------------------------------------------
n_total = len(X_seq)
train_end = int(n_total * 0.6)
val_end = int(n_total * 0.8)

X_train_seq = X_seq[:train_end]
y_train_seq = y_seq[:train_end]

X_val_seq = X_seq[train_end:val_end]
y_val_seq = y_seq[train_end:val_end]

X_test_seq = X_seq[val_end:]
y_test_seq = y_seq[val_end:]

print("\n--- Partición del dataset (No. filas, No. de secuencia temporal de 10 min, No. de características) ---")
print(f"Entrenamiento: {X_train_seq.shape}, Validación: {X_val_seq.shape}, Prueba: {X_test_seq.shape}")

# -------------------------------------------------------------------------
# 5. DEFINIR EL MODELO LSTM EN PyTorch
# -------------------------------------------------------------------------
# 5.1 Comprobar hardware para poder iniciar el entrenamiento del modelo (bien por GPU o CPU)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Usando dispositivo:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Usando dispositivo:", device)

# 5.2 Establecer la clase del modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))  # out: [batch, seq_len, hidden_dim]
        out = out[:, -1, :]              # Último paso: [batch, hidden_dim]
        out = self.fc(out)               # [batch, 1]
        return out

# 5.3 Definir el modelo LSTM
input_dim = len(important_feats)
model = LSTMModel(input_dim, hidden_dim=128, num_layers=3, dropout=0.1).to(device)

# 5.4 Ajustes adicionales del modelo LSTM
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# -------------------------------------------------------------------------
# 6. ENTRENAMIENTO DE LA LSTM USANDO DataLoader (para evitar los molestos errores de OOM)
# -------------------------------------------------------------------------
# 6.1 Tensores
def to_tensor(ndarr):
    return torch.tensor(ndarr, dtype=torch.float32).to(device)

# 6.2 Función auxiliar para la inferencia por batches
def predict_in_batches(model, X, batch_size=64):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size]
            preds.append(model(xb).cpu().detach().numpy())
    return np.concatenate(preds).flatten()

# 6.3 Convertir secuencias de entrenamiento a tensores
X_train_t = to_tensor(X_train_seq)
y_train_t = to_tensor(y_train_seq).view(-1, 1)
X_val_t = to_tensor(X_val_seq)
y_val_t = to_tensor(y_val_seq).view(-1, 1)
X_test_t = to_tensor(X_test_seq)
y_test_t = to_tensor(y_test_seq).view(-1, 1)

# 6.4 Crear un Dataset y DataLoader para entrenamiento
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 6.5 Ajuste del batch_size (valor óptimo: 32 o 64)
train_ds = SequenceDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# 6.6 Ajuste de las épocas de entrenamiento (Es muy importante, a mayor época mejor precisión del modelo)
n_epochs = 5
print("\nIniciando entrenamiento LSTM...")

# 6.7 Inicio del entrenamiento
train_start_time = time.time()
best_val_loss = float('inf')
patience = 3
counter = 0

train_losses = []
val_losses = []

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred_b = model(xb)
        loss_b = criterion(pred_b, yb)
        loss_b.backward()
        optimizer.step()
        epoch_loss += loss_b.item()
    
    # 6.7.1 Validación en batches
    model.eval()
    val_loss = 0.0
    val_batches = 0
    with torch.no_grad():
        for i in range(0, len(X_val_t), 64):
            xb_val = X_val_t[i : i + 64]
            yb_val = y_val_t[i : i + 64]
            preds_val = model(xb_val)
            val_loss += criterion(preds_val, yb_val).item()
            val_batches += 1
    val_loss /= val_batches

    train_loss_avg = epoch_loss / len(train_loader)
    train_losses.append(train_loss_avg)
    val_losses.append(val_loss)
    
    print(f"Época [{epoch+1}/{n_epochs}], Pérdida en entrenamiento: {train_loss_avg:.8f}, Pérdida en validación: {val_loss:.8f}")

    # 6.7.2 Implementar: Early stopping (Evitar el sobre-entrenamiento del modelo vasado en el MSE)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
    if counter >= patience:
        print("Early stopping activado.")
        break

train_time = time.time() - train_start_time
print(f"\nTiempo de entrenamiento LSTM: {train_time:.2f} seg")

# 6.8 Generar gráfica de las pérdidas durante el entrenamiento y la validación
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Pérdidas en entrenamiento')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Pérdidas en validación')
plt.xlabel('EPOCA')
plt.ylabel('MSE')
plt.title('Evolución del error cometido por el modelo LSTM callout durante el entrenamiento')
plt.legend()
plt.grid(True)
plt.savefig('evolucion_MSE_callout_002_LS24-E3-CO3-NC128.png', dpi=900, bbox_inches='tight')
plt.close()
print("Gráfica de evolución del loss guardada como 'evolucion_MSE_callout_002_LS24-E3-CO3-NC128.png'.")

# -------------------------------------------------------------------------
# 6.9 EXPORTAR EL MODELO LSTM
# -------------------------------------------------------------------------
torch.save(model.state_dict(), "LSTM_callout_001.pth")
print("\nModelo LSTM entrenado y exportado como 'LSTM_callout_001.pth'.")

# -------------------------------------------------------------------------
# 7. VALIDACIÓN y TEST DEL MODELO LSTM (Con medición de tiempo de inferencia)
# -------------------------------------------------------------------------
model.eval()

# 7.1 Inferencia en entrenamiento (por batches para evitar los errores de OOM)
train_start_time = time.time()
train_pred_scaled = predict_in_batches(model, X_train_t, batch_size=64)
train_inference_time = time.time() - train_start_time
# Calcular métricas directamente en la escala normalizada
mse_train = mean_squared_error(y_train_t.cpu().numpy().flatten(), train_pred_scaled.flatten())
mae_train = mean_absolute_error(y_train_t.cpu().numpy().flatten(), train_pred_scaled.flatten())
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train_t.cpu().numpy().flatten(), train_pred_scaled.flatten())

# 7.2 Inferencia en validación (por batches)
val_start_time = time.time()
val_pred_scaled = predict_in_batches(model, X_val_t, batch_size=64)
val_inference_time = time.time() - val_start_time
# Calcular métricas directamente en la escala normalizada
mse_val = mean_squared_error(y_val_t.cpu().numpy().flatten(), val_pred_scaled.flatten())
mae_val = mean_absolute_error(y_val_t.cpu().numpy().flatten(), val_pred_scaled.flatten())
rmse_val = np.sqrt(mse_val)
r2_val = r2_score(y_val_t.cpu().numpy().flatten(), val_pred_scaled.flatten())

# 7.3 Inferencia en prueba (por batches)
test_start_time = time.time()
test_pred_scaled = predict_in_batches(model, X_test_t, batch_size=64)
test_inference_time = time.time() - test_start_time
# Calcular métricas directamente en la escala normalizada
mse_test = mean_squared_error(y_test_t.cpu().numpy().flatten(), test_pred_scaled.flatten())
mae_test = mean_absolute_error(y_test_t.cpu().numpy().flatten(), test_pred_scaled.flatten())
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test_t.cpu().numpy().flatten(), test_pred_scaled.flatten())

# 7.4 (Opcional) Calcular media y rango de callout en prueba para el contexto de MAE y RMSE
mean_callout = np.mean(y_test_seq)
min_callout = np.min(y_test_seq)
max_callout = np.max(y_test_seq)

print("\n--- Rendimiento en Entrenamiento (LSTM, escala normalizada) ---")
print(f"MSE: {mse_train:.8f}")
print(f"MAE: {mae_train:.8f}")
print(f"RMSE: {rmse_train:.8f}")
print(f"R^2: {r2_train:.8f}")
print(f"Tiempo de inferencia: {train_inference_time:.2f} seg")

print("\n--- Rendimiento en Validación (LSTM, escala normalizada) ---")
print(f"MSE: {mse_val:.8f}")
print(f"MAE: {mae_val:.8f}")
print(f"RMSE: {rmse_val:.8f}")
print(f"R^2: {r2_val:.8f}")
print(f"Tiempo de inferencia: {val_inference_time:.2f} seg")

print("\n--- Rendimiento en Test (LSTM, escala normalizada) ---")
print(f"MSE: {mse_test:.8f}")
print(f"MAE: {mae_test:.8f}")
print(f"RMSE: {rmse_test:.8f}")
print(f"R^2: {r2_test:.8f}")
print(f"Tiempo de inferencia: {test_inference_time:.2f} seg")
print(f"\n--- Para las estadísticas de callout en Prueba (escala normalizada) ---")
print(f"Media de callout: {mean_callout:.8f}")
#Se comenta la salida del reango pues no es de interés en la escala normalizada, se deja la implementación.
#print(f"Rango de callout: [{min_callout:.8f}, {max_callout:.8f}]") 

# -------------------------------------------------------------------------
# 8. GENERAR 2 SUBFIGURAS (2016 PUNTOS + ZOOM) SUPERPONIENDO 3 TRAMOS DE PREDICCIÓN
# -------------------------------------------------------------------------
print("\nGenerando figura con 2016 puntos y zoom, superponiendo entrenamiento/validación/prueba LSTM...")

# 8.1 Predicción total en la secuencia para la LSTM (por batches)
X_seq_tensor = to_tensor(X_seq)
all_preds_scaled = predict_in_batches(model, X_seq_tensor, batch_size=64)

# 8.1.1 Debido a que creamos secuencias, las primeras seq_len filas no generan predicción
df_seq = df.iloc[seq_len:].copy()
df_seq.reset_index(drop=True, inplace=True)
df_seq['pred_all_scaled'] = all_preds_scaled

# Desescalar las predicciones para las gráficas y mostrar en escala real
df_seq['pred_all'] = target_scaler.inverse_transform(df_seq['pred_all_scaled'].values.reshape(-1, 1)).flatten()

# 8.1.2 Crear 3 columnas para distinguir las predicciones de entrenamiento, validación y prueba
n_all = len(X_seq)
train_end_seq = int(n_all * 0.6)
val_end_seq = int(n_all * 0.8)

df_seq['pred_train'] = 0.0
df_seq['pred_val']   = 0.0
df_seq['pred_test']  = 0.0

df_seq.loc[:train_end_seq-1, 'pred_train'] = df_seq.loc[:train_end_seq-1, 'pred_all']
df_seq.loc[train_end_seq:val_end_seq-1, 'pred_val'] = df_seq.loc[train_end_seq:val_end_seq-1, 'pred_all']
df_seq.loc[val_end_seq:, 'pred_test'] = df_seq.loc[val_end_seq:, 'pred_all']

# 8.1.3 Reconstruir 'time_bin'
df_seq['time_bin'] = (pd.to_datetime(df_seq['date'].astype(str))
                      + pd.to_timedelta(df_seq['hour'], unit='h')
                      + pd.to_timedelta(df_seq['minute'], unit='m'))

# 8.1.4 Agrupar en intervalos de 10 min. Se asume que el agrupamiento da 2016 intervalos (14 días)
df_plot = df_seq.groupby('time_bin', as_index=False).agg({
    'callout': 'sum',
    'pred_train': 'sum',
    'pred_val': 'sum',
    'pred_test': 'sum'
})
y_real = df_plot['callout'].values
y_tr   = df_plot['pred_train'].values
y_vl   = df_plot['pred_val'].values
y_ts   = df_plot['pred_test'].values

print(f"df_plot total de filas: {len(df_plot)} (total de intervalos de 10 min).")

# 8.2 Función para calcular SMAPE para un resultado entre el 0 % y el 200 %
# (En la práctica, para resultados entre el 0 % y el 100 % se elimina el factor 0.5 en denominator)
def calculate_smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) # / 2
    diff = np.abs(y_pred - y_true) / denominator
    return 100 * np.mean(diff)

# 8.3 Combinar predicciones para los 2016 puntos
y_pred_all_grouped = (df_plot['pred_train'].fillna(0) + 
                      df_plot['pred_val'].fillna(0) + 
                      df_plot['pred_test'].fillna(0)).values

# 8.4 Calcular SMAPE total (2016 puntos)
smape_total = calculate_smape(y_real, y_pred_all_grouped)

# 8.5 Calcular SMAPE para el zoom [1728:1872]
zoom_start = 1728
zoom_end = 1872
y_real_zoom = y_real[zoom_start:zoom_end]
y_ts_zoom = y_ts[zoom_start:zoom_end]
smape_zoom = calculate_smape(y_real_zoom, y_ts_zoom)

# 8.6 Generar gráfica con dos subfiguras
import math
fig_width_pt = 345
inches_per_pt = 1.0 / 72.27
golden_mean = (math.sqrt(5) - 1.0) / 2.0
fig_width = fig_width_pt * inches_per_pt
fig_height = fig_width * golden_mean

sns.set_style("ticks")
sns.set_context("paper")

fig, axs = plt.subplots(2, 1, figsize=(fig_width, fig_height * 2))

# Subfigura 1: 2016 intervalos (14 días), mostrando real y 3 tramos de predicción
axs[0].plot(y_real,  label='Callout', linewidth=0.5, color='black')
axs[0].plot(y_tr,    label='Entrenamiento', linewidth=0.5, color='red')
axs[0].plot(y_vl,    label='Validación',   linewidth=0.5, color='orange')
axs[0].plot(y_ts,    label='Prueba',       linewidth=0.5, color='green')
axs[0].set_xticks(np.arange(0, len(y_real), step=144))
axs[0].tick_params(axis='x', labelsize=8, rotation=45)
axs[0].tick_params(axis='y', labelsize=8)
axs[0].legend(loc='center right', fontsize=8)
sns.despine(ax=axs[0])
axs[0].set_xlabel("Intervalos de 10 minutos (2016 en 14 días)", fontsize=10)
axs[0].set_ylabel("Total de callout", fontsize=10)
axs[0].set_title(f"Tráfico real vs. predicción LSTM \nSMAPE total: {smape_total:.2f}%", fontsize=12)

# Subfigura 2: Zoom en [1728:1872] (1 día), solo predicción en prueba
x_zoom = np.arange(zoom_start, zoom_end)
axs[1].plot(x_zoom, y_real[zoom_start:zoom_end],
            label='Tráfico real (callout)',
            linewidth=0.8, marker='.', markersize=3, color='blue')
axs[1].plot(x_zoom, y_ts[zoom_start:zoom_end],
            label='Predicción (prueba)',
            linewidth=0.8, marker='.', markersize=3, color='green')
axs[1].set_xticks(np.arange(zoom_start, zoom_end, step=6))
axs[1].tick_params(axis='x', labelsize=8, rotation=45)
axs[1].tick_params(axis='y', labelsize=8)
axs[1].legend(loc='center right', fontsize=8)
sns.despine(ax=axs[1])
axs[1].set_xlabel("Intervalos de 10 minutos (Zoom: 1 día)", fontsize=10)
axs[1].set_ylabel("Total de callout", fontsize=10)
axs[1].set_title(f"Tráfico real vs. predicción LSTM callout \nSMAPE zoom: {smape_zoom:.2f}%", fontsize=12)

plt.tight_layout()
plt.savefig("Integracion_LightGBM_LSTM_callout_002_LS24-E3-CO3-NC128.png", dpi=900, bbox_inches='tight')
plt.close()

print("Figura 'Integracion_LightGBM_LSTM_callout_002_LS24-E3-CO3-NC128.png' creada.")
print("Fin del script.")