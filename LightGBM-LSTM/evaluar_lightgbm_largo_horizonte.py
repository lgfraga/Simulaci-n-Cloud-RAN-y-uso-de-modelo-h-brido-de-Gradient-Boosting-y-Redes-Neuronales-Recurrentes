#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluación “long‑horizon” para el modelo LightGBM_callout_001.pkl

* Repite exactamente la ingeniería de características usada al entrenar.
* Agrega las predicciones y los valores reales por intervalo de 10 min.
* Devuelve (guarda) las series alineadas y el SMAPE en un fichero NPZ.
"""

import argparse, math, time
import numpy as np
import pandas as pd
import joblib, lightgbm as lgb
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------- argumentos CLI
parser = argparse.ArgumentParser(
    description="Evalúa LightGBM call‑out en un horizonte de N días.")
parser.add_argument("--days", type=int, default=1,
                    help="Horizonte en días (default: 1).")
args = parser.parse_args()
HORIZON_DAYS = args.days
HORIZON_ITVS = 144 * HORIZON_DAYS      # 144 intervalos de 10 min por día

# -------------------------------------------------- constantes ruta
DATA_CSV   = "dataset_callout.csv"
MODEL_PKL  = "LightGBM_callout_001.pkl"
SCALER_PKL = "scaler_callout_002.pkl"   # escalador del target
OUT_NPZ    = f"lightgbm_long_{HORIZON_DAYS}d.npz"

# -------------------------------------------------- utilidades
def add_cyclic_features(df):
    df['hour_sin']     = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos']     = np.cos(2*np.pi*df['hour']/24)
    df['weekday_sin']  = np.sin(2*np.pi*df['weekday']/7)
    df['weekday_cos']  = np.cos(2*np.pi*df['weekday']/7)
    return df

def calc_smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    # evitar división 0/0
    return 100*np.mean(np.where(denom==0, 0, np.abs(y_true-y_pred)/denom))

# -------------------------------------------------- carga y prep datos
print("[LightGBM] cargando dataset call‑out …")
df = pd.read_csv(DATA_CSV)
df.sort_values(['date','hour','minute','CellID'], inplace=True)
df.reset_index(drop=True, inplace=True)

# target escalado (MISMO escalador que el entrenamiento)
target_scaler = joblib.load(SCALER_PKL)
df['callout_scaled'] = target_scaler.transform(df[['callout']])

# features cíclicas + lags
df = add_cyclic_features(df)
df['lag1'] = df['callout_scaled'].shift(1)
df['lag2'] = df['callout_scaled'].shift(2)
df.dropna(subset=['lag1','lag2'], inplace=True)

X = df[['hour_sin','hour_cos',
        'weekday_sin','weekday_cos',
        'lag1','lag2','idx','CellID']].copy()
X['CellID'] = X['CellID'].astype('category')
y_real_scaled = df['callout_scaled'].values

# -------------------------------------------------- predicción
print("[LightGBM] cargando modelo …")
model: lgb.Booster = joblib.load(MODEL_PKL)
y_pred_scaled = model.predict(X)

# des‑escalado a valores reales
y_real = target_scaler.inverse_transform(y_real_scaled.reshape(-1,1)).flatten()
y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()

# -------------------------------------------------- agregación 10 min
df['time_bin'] = (pd.to_datetime(df['date'].astype(str)) +
                  pd.to_timedelta(df['hour'], unit='h') +
                  pd.to_timedelta(df['minute'], unit='m'))

df['pred'] = y_pred
agg = df.groupby('time_bin', as_index=False).agg(
        real=('callout','sum'),
        pred=('pred','sum'))

# -------------------------------------------------- recorte horizonte
data_slice = agg.tail(HORIZON_ITVS)
y_real_slice = data_slice['real'].values
y_pred_slice = data_slice['pred'].values
smape_total  = calc_smape(y_real_slice, y_pred_slice)

print(f"[LightGBM] SMAPE ({HORIZON_DAYS} días) = {smape_total:.2f}%")
np.savez_compressed(OUT_NPZ,
                    time=data_slice['time_bin'].values.astype('datetime64[s]'),
                    real=y_real_slice,
                    lgbm=y_pred_slice,
                    smape=smape_total)
print(f"[LightGBM] guardado {OUT_NPZ}")
