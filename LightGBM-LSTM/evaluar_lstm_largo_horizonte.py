#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluación “long‑horizon” para el modelo LSTM_callout_002.pth

* Reproduce la misma selección de features y el mismo escalado que en
  el entrenamiento.
* Predice toda la serie (secuencia‑a‑uno) y agrega por intervalo.
* Devuelve valores reales, predicción LSTM y SMAPE en un NPZ.
"""

import argparse, math, time, warnings
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import joblib
from pathlib import Path
warnings.filterwarnings("ignore")

# ----------------------------------------------- argumentos CLI
parser = argparse.ArgumentParser(
    description="Evalúa LSTM call‑out en un horizonte de N días.")
parser.add_argument("--days", type=int, default=1,
                    help="Horizonte en días (default: 1).")
args = parser.parse_args()
HORIZON_DAYS  = args.days
HORIZON_ITVS  = 144*HORIZON_DAYS
SEQ_LEN       = 24

# ----------------------------------------------- rutas
DATA_CSV   = "dataset_callout.csv"
MODEL_PTH  = "LSTM_callout_002.pth"
SCAL_FEAT  = "scaler_callout_features_001.pkl"
SCAL_TGT   = "scaler_callout_target_001.pkl"
OUT_NPZ    = f"lstm_long_{HORIZON_DAYS}d.npz"

important_feats = ['hour_sin','hour_cos',
                   'lag1','lag2',
                   'weekday_sin','weekday_cos',
                   'lag3','lag4']

# ----------------------------------------------- utilidades
def add_cyclic(df):
    df['hour_sin']    = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos']    = np.cos(2*np.pi*df['hour']/24)
    df['weekday_sin'] = np.sin(2*np.pi*df['weekday']/7)
    df['weekday_cos'] = np.cos(2*np.pi*df['weekday']/7)
    return df

def calc_smape(y_true, y_pred):
    denom = np.abs(y_true)+np.abs(y_pred)
    return 100*np.mean(np.where(denom==0,0,np.abs(y_true-y_pred)/denom))

# ----------------------------------------------- modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers>1 else 0.0)
        self.fc   = nn.Linear(hidden_dim,1)
    def forward(self,x):
        out,_ = self.lstm(x)
        return self.fc(out[:,-1,:])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------- carga & prep datos
print("[LSTM] cargando dataset …")
df = pd.read_csv(DATA_CSV)
df.sort_values(['date','hour','minute','CellID'], inplace=True)
df.reset_index(drop=True, inplace=True)

# escalado objetivo
scaler_target = joblib.load(SCAL_TGT)
df['callout_scaled'] = scaler_target.transform(df[['callout']])

# features
df = add_cyclic(df)
for lag in (1,2,3,4):
    df[f'lag{lag}'] = df['callout_scaled'].shift(lag)
df.dropna(inplace=True)

# normalizar features con escalador entrenado
scaler_feat = joblib.load(SCAL_FEAT)
df[important_feats] = scaler_feat.transform(df[important_feats])

feat_arr = df[important_feats].values.astype(np.float32)
target_arr = df['callout_scaled'].values.astype(np.float32)

# ----------------------------------------------- secuencias
X_seq = np.array([feat_arr[i:i+SEQ_LEN] for i in range(len(feat_arr)-SEQ_LEN)])
y_seq = target_arr[SEQ_LEN:]

# ----------------------------------------------- modelo
print("[LSTM] cargando modelo …")
model = LSTMModel(len(important_feats)).to(device)
model.load_state_dict(torch.load(MODEL_PTH, map_location=device))
model.eval()

@torch.no_grad()
def predict_batches(model, X, bs=64):
    preds=[]
    for i in range(0,len(X),bs):
        x = torch.tensor(X[i:i+bs]).to(device)
        preds.append(model(x).cpu().numpy())
    return np.vstack(preds).flatten()

print("[LSTM] inferencia …")
y_pred_scaled = predict_batches(model, X_seq)
# des‑escalado
y_pred = scaler_target.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
y_real = scaler_target.inverse_transform(y_seq.reshape(-1,1)).flatten()

# align índices con df después de SEQ_LEN
df_seq = df.iloc[SEQ_LEN:].copy()
df_seq['pred'] = y_pred

# agregación 10 min
df_seq['time_bin'] = (pd.to_datetime(df_seq['date'].astype(str)) +
                      pd.to_timedelta(df_seq['hour'],unit='h') +
                      pd.to_timedelta(df_seq['minute'],unit='m'))
agg = df_seq.groupby('time_bin', as_index=False).agg(
        real=('callout','sum'), pred=('pred','sum'))

# recorte horizonte final
slice_df = agg.tail(HORIZON_ITVS)
y_real_slice = slice_df['real'].values
y_pred_slice = slice_df['pred'].values
smape_total  = calc_smape(y_real_slice, y_pred_slice)

print(f"[LSTM] SMAPE ({HORIZON_DAYS} días) = {smape_total:.2f}%")
np.savez_compressed(OUT_NPZ,
                    time=slice_df['time_bin'].values.astype('datetime64[s]'),
                    real=y_real_slice,
                    lstm=y_pred_slice,
                    smape=smape_total)
print(f"[LSTM] guardado {OUT_NPZ}")
