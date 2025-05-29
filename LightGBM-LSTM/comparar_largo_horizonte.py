#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combina los NPZ de LightGBM y LSTM y dibuja en una sola figura:
  * valores reales
  * predicción LightGBM
  * predicción LSTM
Incluye los SMAPE totales en el título.
"""

import argparse, math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--days",type=int,default=1,help="días evaluados (default 1)")
args = parser.parse_args()
HORIZON_DAYS = args.days

lgb_npz  = np.load(f"lightgbm_long_{HORIZON_DAYS}d.npz")
lstm_npz = np.load(f"lstm_long_{HORIZON_DAYS}d.npz")

assert (lgb_npz['time']==lstm_npz['time']).all(), "Las series no están alineadas"

time_axis = np.arange(len(lgb_npz['time']))
y_real    = lgb_npz['real']
y_lgbm    = lgb_npz['lgbm']
y_lstm    = lstm_npz['lstm']
smape_lgb = float(lgb_npz['smape'])
smape_lstm= float(lstm_npz['smape'])

# -------------------------------------------------- gráfica
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(time_axis, y_real, label="Real", color="black", linewidth=.6)
ax.plot(time_axis, y_lgbm, label=f"LightGBM (SMAPE {smape_lgb:.2f}%)",
        color="red", linewidth=.6)
ax.plot(time_axis, y_lstm, label=f"LSTM (SMAPE {smape_lstm:.2f}%)",
        color="green", linewidth=.6)

step_xticks = 6 if HORIZON_DAYS==1 else 144                # 1 h cuando 1 día
ax.set_xticks(np.arange(0, len(time_axis), step=step_xticks))
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.tick_params(axis='y', labelsize=8)
ax.set_xlabel(f"Intervalos de 10 min ({HORIZON_DAYS} días)", fontsize=10)
ax.set_ylabel("Total callout", fontsize=10)
ax.set_title(f"Comparación LightGBM vs LSTM — Horizonte {HORIZON_DAYS} día(s)",
             fontsize=12)
ax.legend(loc="best", fontsize=8)
sns.despine()
plt.tight_layout()
fname = f"comparacion_callout_{HORIZON_DAYS}d.png"
plt.rcParams['axes.unicode_minus'] = False
plt.savefig(fname, dpi=300, bbox_inches="tight")
print(f"[COMPARE] figura guardada → {fname}")
