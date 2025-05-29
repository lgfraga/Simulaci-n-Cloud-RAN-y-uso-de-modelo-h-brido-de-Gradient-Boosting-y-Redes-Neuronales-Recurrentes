#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------- AUTOR: Luis González Fraga  ----------------
#---------------------------  BREVE DESCRIPCIÓN DEL SCRIPT  ---------------
# -------------------------------------------------------------------------
# 1. CARGAR EL DATASET
# -------------------------------------------------------------------------
# Esta sección carga el dataset desde el archivo "dataset_smsout.csv" y lo ordena temporalmente 
# basado en las columnas 'date', 'hour', 'minute' y 'CellID'. La ordenación cronológica es necesaria 
# para el análisis de series temporales, ya que asegura que los datos se procesen en la secuencia 
# correcta, lo cual es fundamental para capturar patrones y dependencias temporales en las predicciones.

# -------------------------------------------------------------------------
# 2. CREAR CARACTERÍSTICAS CÍCLICAS + LAGS
# -------------------------------------------------------------------------
# En esta sección, se añaden las características cíclicas para poder capturar los patrones periódicos en los datos, 
# como la hora del día y el día de la semana, utilizando las transformaciones de seno y coseno. Estas 
# transformaciones permiten que el modelo interprete la naturaleza cíclica de estas variables. 
# Además, se incorporan lags (retardos) de la variable objetivo que es 'smsout' (específicamente, lag1 y lag2), 
# lo que permite al modelo considerar los valores pasados de la serie temporal para mejorar las predicciones.

# -------------------------------------------------------------------------
# 3. DEFINIR X e Y
# -------------------------------------------------------------------------
# Se define la variable objetivo 'Y' como la columna 'smsout', que es el valor que se desea predecir. 
# Las características 'X' incluyen las características cíclicas (hour_sin, hour_cos, weekday_sin, weekday_cos), 
# los lags (lag1, lag2), y otras columnas relevantes como 'idx' y 'CellID'. La columna 'CellID' se convierte 
# a un tipo categórico para que el modelo LightGBM pueda manejarla adecuadamente durante el entrenamiento.

# -------------------------------------------------------------------------
# 4. DIVISIÓN DEL DATASET: ENTRENAMIENTO (60%), VALIDACIÓN (20%) Y PRUEBA (20%)
# -------------------------------------------------------------------------
# El dataset se divide en tres conjuntos: entrenamiento (60%), validación (20%) y prueba (20%). 
# Esta división es necesaria para entrenar el modelo con los datos de entrenamiento, ajustar los 
# hiperparámetros utilizando el conjunto de validación, y evaluar el rendimiento final del modelo 
# en el conjunto de prueba de manera imparcial. La división se realiza manteniendo el orden temporal 
# de los datos, lo cual es importante para series temporales.

# -------------------------------------------------------------------------
# 5. AJUSTES BÁSICO DE HIPERPARÁMETROS
# -------------------------------------------------------------------------
# En esta sección, se realiza un ajuste básico de los hiperparámetros para el modelo LightGBM utilizando 
# RandomizedSearchCV. Se define un conjunto de posibles valores para los parámetros clave como num_leaves, 
# max_depth, learning_rate, entre otros. Para acelerar el proceso, se utiliza un subconjunto de los 
# datos (hasta 200,000). 
# Se emplea un split predefinido para mantener la consistencia en la división de los datos durante la 
# búsqueda de hiperparámetros. Una vez encontrados los mejores parámetros, el modelo se entrena nuevamente 
# utilizando todos los datos de entrenamiento y validación.

# -------------------------------------------------------------------------
# 6. EVALUACIÓN EN PRUEBA
# -------------------------------------------------------------------------
# Después de entrenar el modelo con los mejores hiperparámetros, se evalúa su rendimiento en el conjunto 
# de prueba. Se calculan métricas de error como MSE (Mean Squared Error), MAE (Mean Absolute Error), 
# RMSE (Root Mean Squared Error), R^2 (coeficiente de determinación) para medir la precisión de las predicciones. 
# Además, se mide el tiempo de inferencia para evaluar la eficiencia del modelo en la predicción de nuevos datos.

# -------------------------------------------------------------------------
# 7. GRAFICAR EN 2 SUBFIGURAS (2016 PUNTOS + ZOOM)
# -------------------------------------------------------------------------
# Esta sección genera dos subfiguras para visualizar los resultados:
# Subfigura 1: Muestra la predicción completa sobre 2016 intervalos de tiempo (correspondientes a 14 días), 
# superponiendo los tramos de entrenamiento, validación y prueba junto con los valores reales de 'smsout'.
# Subfigura 2: Realiza un zoom en un rango específico (índices 1728 a 1872, equivalente a 1 día) para 
# visualizar detalladamente la predicción en el conjunto de prueba.
# En ambas subfiguras, se calcula y muestra el valor de SMAPE (symmetric mean absolute percentage error)
# que es una medida adicional de la precisión del modelo pero sin aplicar el factor 0.5 en el denominador 
# de la ecuación, para proporcionar una interpretación mucho más fácil del error porcentual en un rango 
# entre el 0% y el 100%.
# SMAPE < 10% = excelente
# 10% < SMAPE < 20% = válido
# SMAPE > 20% = inadecuado
# FUENTE: https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMRegressor
import joblib

# -------------------------------------------------------------------------
# 1. CARGAR EL DATASET
# -------------------------------------------------------------------------
df = pd.read_csv("dataset_smsout.csv")
print(f"Total de filas en dataset_smsout.csv: {len(df)}")

# Orden temporal
df.sort_values(by=['date', 'hour', 'minute', 'CellID'], inplace=True)
df.reset_index(drop=True, inplace=True)

# -------------------------------------------------------------------------
# 2. NORMALIZAR LOS DATOS DE 'smsout'
# -------------------------------------------------------------------------
scaler = MinMaxScaler()
df['smsout_scaled'] = scaler.fit_transform(df[['smsout']])

# -------------------------------------------------------------------------
# 3. CREAR CARACTERÍSTICAS CÍCLICAS + LAGS (usando la versión escalada)
# -------------------------------------------------------------------------
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)

df['lag1'] = df['smsout_scaled'].shift(1)
df['lag2'] = df['smsout_scaled'].shift(2)

df.dropna(subset=['lag1', 'lag2'], inplace=True)

# -------------------------------------------------------------------------
# 4. DEFINIR X e Y (usando la versión escalada)
# -------------------------------------------------------------------------
y = df['smsout_scaled'].values
X = df[['hour_sin', 'hour_cos',
        'weekday_sin', 'weekday_cos',
        'lag1', 'lag2',
        'idx', 'CellID']].copy()
X['CellID'] = X['CellID'].astype('category')

# -------------------------------------------------------------------------
# 5. DIVISIÓN DEL DATASET: ENTRENAMIENTO (60%), VALIDACIÓN (20%) Y PRUEBA (20%)
# -------------------------------------------------------------------------
n = len(df)
train_end = int(n * 0.6)  # train hasta 60%
val_end = int(n * 0.8)    # val hasta 80%, test el 20% final

X_train = X.iloc[:train_end]
y_train = y[:train_end]

X_val = X.iloc[train_end:val_end]
y_val = y[train_end:val_end]

X_test = X.iloc[val_end:]
y_test = y[val_end:]
print(f"\n(No. Filas del dataset, No. de características)")
print(f"Entrenamiento: {X_train.shape}, Validación: {X_val.shape}, Prueba: {X_test.shape}")

# -------------------------------------------------------------------------
# 6. AJUSTES BÁSICO DE HIPERPARÁMETROS
# -------------------------------------------------------------------------
sample_train_val = 200000
tv_end = val_end
if tv_end > sample_train_val:
    X_tune = X.iloc[:sample_train_val]
    y_tune = y[:sample_train_val]
else:
    X_tune = X.iloc[:tv_end]
    y_tune = y[:tv_end]

split_index = np.ones(len(X_tune), dtype=int) * 0  # 0 => train fold
split_index[int(len(X_tune) * 0.6):int(len(X_tune) * 0.8)] = -1  # val fold
split_index[int(len(X_tune) * 0.8):] = -1
pds = PredefinedSplit(test_fold=split_index)

param_dist = {
    'num_leaves': [15, 31, 63, 127],
    'max_depth': [3, 5, 7, -1],
    'learning_rate': [0.1, 0.05, 0.01],
    'n_estimators': [300, 600, 1000],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0, 0.1, 1]
}

lgbm_estimator = LGBMRegressor(
    objective='regression',
    random_state=42,
    verbose=-1  # Silencia mensajes "No further splits..."
)

rs = RandomizedSearchCV(
    estimator=lgbm_estimator,
    param_distributions=param_dist,
    n_iter=10,
    scoring='neg_mean_squared_error',
    cv=pds,
    verbose=1,
    random_state=42
)

print("\nIniciando RandomizedSearchCV (Ajuste básico de hiperparámetros)...")
rs.fit(X_tune, y_tune)
print("Mejores parámetros:", rs.best_params_)
print("Mejor puntuación (neg MSE):", rs.best_score_)

best_model = rs.best_estimator_
print("\nEntrenando modelo con todos los datos de Entrenamiento y validación, con los mejores parámetros encontrados...")

X_tv = X.iloc[:val_end]
y_tv = y[:val_end]
best_model.fit(X_tv, y_tv)

joblib.dump(best_model, "LightGBM_smsout_001.pkl")
print("\nSerializando y exportando el modelo entrenado: LightGBM_smsout_001.pkl ...")

# Exportar el escalador después de ajustarlo a los datos de entrenamiento
joblib.dump(scaler, "scaler_smsout_001.pkl")
print("Escalador exportado como 'scaler_smsout_001.pkl'.")

# -------------------------------------------------------------------------
# 7. EVALUACIÓN EN PRUEBA (en escala normalizada)
# -------------------------------------------------------------------------
test_start_time = time.time()
y_test_pred_scaled = best_model.predict(X_test)
test_inference_time = time.time() - test_start_time

mse_test = mean_squared_error(y_test, y_test_pred_scaled)
mae_test = mean_absolute_error(y_test, y_test_pred_scaled)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_test_pred_scaled)

# Función clásica para SMAPE (resultado entre el 0 % y el 200 %)
# (En la práctica, para resultados entre el 0 % y el 100 % se elimina el factor 0.5 en denominator)
def calculate_smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) # / 2
    diff = np.abs(y_pred - y_true) / denominator
    return 100 * np.mean(diff)


smape_test = calculate_smape(y_test, y_test_pred_scaled)

# Calcular media.
mean_smsout = np.mean(y_test)

print(f"\n--- Rendimiento final en Test (escala normalizada) ---")
print(f"MSE: {mse_test:.8f}")
print(f"MAE: {mae_test:.8f}")
print(f"RMSE: {rmse_test:.8f}")
print(f"R^2: {r2_test:.8f}")
print(f"SMAPE en Test (utilizando la ecuación práctica): {smape_test:.8f}%")
print(f"Media de smsout: {mean_smsout:.8f}")
print(f"Tiempo de inferencia: {test_inference_time:.2f} seg")

# -------------------------------------------------------------------------
# 8. GRAFICAR EN 2 SUBFIGURAS (2016 PUNTOS + ZOOM)
# -------------------------------------------------------------------------
print("\nGenerando la figura con 2016 puntos y el zoom [1728:1872], con 3 tramos de predicción...")

# 8.1 Predicciones totales (escala normalizada)
y_pred_all_scaled = best_model.predict(X)
df['pred_all_scaled'] = y_pred_all_scaled

# Crear 3 columnas de predicción: pred_train, pred_val, pred_test (escala normalizada)
df['pred_train_scaled'] = np.nan
df['pred_val_scaled'] = np.nan
df['pred_test_scaled'] = np.nan

df.loc[:train_end-1, 'pred_train_scaled'] = df.loc[:train_end-1, 'pred_all_scaled']
df.loc[train_end:val_end-1, 'pred_val_scaled'] = df.loc[train_end:val_end-1, 'pred_all_scaled']
df.loc[val_end:, 'pred_test_scaled'] = df.loc[val_end:, 'pred_all_scaled']

# 8.2 Desescalar las predicciones para la visualización
df['pred_train'] = scaler.inverse_transform(df[['pred_train_scaled']].fillna(0))
df['pred_val'] = scaler.inverse_transform(df[['pred_val_scaled']].fillna(0))
df['pred_test'] = scaler.inverse_transform(df[['pred_test_scaled']].fillna(0))

# 8.3 time_bin para agrupar
df['time_bin'] = (
    pd.to_datetime(df['date'].astype(str))
    + pd.to_timedelta(df['hour'], unit='h')
    + pd.to_timedelta(df['minute'], unit='m')
)

# 8.4 Agrupar 'smsout' y cada columna de pred (en escala original)
df_plot = df.groupby('time_bin', as_index=False).agg({
    'smsout': 'sum',
    'pred_train': 'sum',
    'pred_val': 'sum',
    'pred_test': 'sum'
})

y_real = df_plot['smsout'].values
y_train = df_plot['pred_train'].values
y_val = df_plot['pred_val'].values
y_test_ = df_plot['pred_test'].values

print(f"Longitud de df_plot: {len(df_plot)} filas (intervalos).")

# 8.5 Combinar predicciones para los 2016 puntos (en escala original)
y_pred_all_grouped = (df_plot['pred_train'].fillna(0) +
                      df_plot['pred_val'].fillna(0) +
                      df_plot['pred_test'].fillna(0)).values

# 8.6 Calcular SMAPE total (2016 puntos, en escala original)
smape_total = calculate_smape(y_real, y_pred_all_grouped)

# 8.7 Calcular SMAPE para el zoom [1728:1872] (en escala original)
zoom_start = 1728
zoom_end = 1872
y_real_zoom = y_real[zoom_start:zoom_end]
y_test_zoom = y_test_[zoom_start:zoom_end]
smape_zoom = calculate_smape(y_real_zoom, y_test_zoom)

# 8.9 Armado de la figura
import math
fig_width_pt = 345
inches_per_pt = 1.0 / 72.27
golden_mean = (math.sqrt(5) - 1.0) / 2.0
fig_width = fig_width_pt * inches_per_pt
fig_height = fig_width * golden_mean

sns.set_style("ticks")
sns.set_context("paper")

fig, axs = plt.subplots(2, 1, figsize=(fig_width, fig_height * 2))

# Subfigura 1: Muestra tráfico real vs. predicción (entrenamiento, validación y prueba)
axs[0].plot(y_real, label='smsout', linewidth=0.5, color='black')
axs[0].plot(y_train, label='Entrenamiento', linewidth=0.5, color='red')
axs[0].plot(y_val, label='Validación', linewidth=0.5, color='orange')
axs[0].plot(y_test_, label='Prueba', linewidth=0.5, color='green')

axs[0].set_xticks(np.arange(0, len(y_real), step=144))
axs[0].tick_params(axis='x', labelsize=8, rotation=45)
axs[0].tick_params(axis='y', labelsize=8)
axs[0].legend(loc='center right', fontsize=8)
sns.despine(ax=axs[0])
axs[0].set_xlabel("Intervalos de 10 minutos (2016 en 14 días)", fontsize=10)
axs[0].set_ylabel("Total de smsout", fontsize=10)
axs[0].set_title(f"LightGBM - Total de tráfico real vs predicción\nSMAPE total: {smape_total:.2f}%", fontsize=12)

# Subfigura 2: Zoom en [1728:1872] => solo la predicción de prueba
x_zoom = np.arange(zoom_start, zoom_end)
axs[1].plot(x_zoom, y_real[zoom_start:zoom_end],
            label='Tráfico real (smsout)',
            linewidth=0.8, marker='.', markersize=3, color='blue')
axs[1].plot(x_zoom, y_test_[zoom_start:zoom_end],
            label='Predicción (prueba)',
            linewidth=0.8, marker='.', markersize=3, color='green')

axs[1].set_xticks(np.arange(zoom_start, zoom_end, step=6))
axs[1].tick_params(axis='x', labelsize=8, rotation=45)
axs[1].tick_params(axis='y', labelsize=8)
axs[1].legend(loc='center right', fontsize=8)
sns.despine(ax=axs[1])
axs[1].set_xlabel("Intervalos de 10 minutos (Zoom: 1 día)", fontsize=10)
axs[1].set_ylabel("Total de smsout", fontsize=10)
axs[1].set_title(f"Total de tráfico real vs predicción (Prueba)\nSMAPE zoom: {smape_zoom:.2f}%", fontsize=12)

plt.tight_layout()
plt.savefig("Regresion_LightGBM_smsout_001.png", format='png', dpi=330, bbox_inches='tight')
plt.close()

print("Figura 'Regresion_LightGBM_smsout_001.png' creada.")
print("Fin del script.")