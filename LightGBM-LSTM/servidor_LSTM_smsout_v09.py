from fastapi import FastAPI, File, UploadFile
import pandas as pd, numpy as np, torch, torch.nn as nn
import joblib, logging, os, math, time

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------- #
#   Modelo y assets                                               #
# --------------------------------------------------------------- #
class LSTMModel(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        self.h, self.l = hidden_dim, num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=(dropout if num_layers>1 else 0))
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        h0 = torch.zeros(self.l, x.size(0), self.h, device=x.device)
        c0 = torch.zeros_like(h0)
        out,_ = self.lstm(x,(h0,c0))
        return self.fc(out[:, -1, :])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMModel().to(device)
model.load_state_dict(torch.load("LSTM_smsout_001.pth", map_location=device))
model.eval()
feat_scaler = joblib.load("scaler_smsout_features_001.pkl")
tgt_scaler  = joblib.load("scaler_smsout_target_001.pkl")
logger.info("Modelo y escaladores cargados.")

# constantes
FEATS   = ['hour_sin','hour_cos','lag1','lag2','weekday_sin','weekday_cos','lag3','lag4']
SEQLEN  = 24   # 4 h

# --------------------------------------------------------------- #
#   Utilidades                                                    #
# --------------------------------------------------------------- #
def _time_bin(df):
    df["time_bin"] = (pd.to_datetime(df["date"])
                      + pd.to_timedelta(df["hour"],   unit="h")
                      + pd.to_timedelta(df["minute"], unit="m"))
    return df

def _feat_eng(df):
    df["hour_sin"]    = np.sin(2*np.pi*df["hour"]    / 24)
    df["hour_cos"]    = np.cos(2*np.pi*df["hour"]    / 24)
    df["weekday_sin"] = np.sin(2*np.pi*df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2*np.pi*df["weekday"] / 7)
    return df

def _add_scaled_target_lags(df):
    df["smsout_scaled"] = tgt_scaler.transform(df[["smsout"]])
    for k in (1,2,3,4):
        df[f"lag{k}"] = df["smsout_scaled"].shift(k)  # shift global, igual que en el entrenamiento
    return df

def _last_sequences(df24):
    """
    Devuelve un array (n_celdas, SEQLEN, 8 características) con la última
    secuencia de 24 instantes por cada CellID que la tenga completa.
    """
    X = []
    for _, g in df24.groupby("CellID"):
        if len(g) < SEQLEN:               # no llega a 4 h
            continue
        seq = g[FEATS].tail(SEQLEN).values
        X.append(seq)
    return np.asarray(X, dtype=np.float32)

# --------------------------------------------------------------- #
@app.get("/status")
def health():
    return {"status":"ok","msg":"LSTM predictor ready"}

@app.post("/predict/lstm_csv")
async def predict(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(
            pd.io.common.StringIO((await file.read()).decode())
        )
        df = _time_bin(df)
        if df["time_bin"].nunique() != SEQLEN:
            raise ValueError(f"Se requieren exactamente {SEQLEN} intervalos de 10 min.")

        df.sort_values(["time_bin","CellID"], inplace=True)
        df = _feat_eng(df)
        df = _add_scaled_target_lags(df).dropna(subset=[f"lag{k}" for k in (1,2,3,4)])

        df[FEATS] = feat_scaler.transform(df[FEATS])

        X = _last_sequences(df)
        if X.size == 0:
            raise ValueError("Ninguna celda dispone de 24 mediciones completas.")

        Xt = torch.tensor(X, dtype=torch.float32, device=device)

        t0 = time.perf_counter()
        with torch.no_grad():
            pred_scaled = model(Xt).cpu().numpy().flatten()
        inf_t = time.perf_counter() - t0

        pred = tgt_scaler.inverse_transform(pred_scaled.reshape(-1,1)).flatten()
        total_smsout = float(pred.sum())

        next_tb = df["time_bin"].max() + pd.Timedelta(minutes=10)

        return {"status":"success",
                "predicted_total_smsout": total_smsout,
                "inference_time": inf_t,
                "predicted_for": str(next_tb)}
    except Exception as e:
        logger.error(f"LSTM error: {e}")
        return {"status":"error","message":str(e)}

if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="192.168.1.39", port=8003)
