from fastapi import FastAPI, File, UploadFile
import pandas as pd, numpy as np, joblib, logging, os, time

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH  = "LightGBM_internet_001.pkl"
SCALER_PATH = "scaler_internet_001.pkl"
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
logger.info("Modelo y escalador cargados.")

# ---------------------------------------------------------------
def _time_bin(df):
    df["time_bin"] = (
        pd.to_datetime(df["date"])
        + pd.to_timedelta(df["hour"],   unit="h")
        + pd.to_timedelta(df["minute"], unit="m")
    )
    return df

def _add_features(df):
    df["hour_sin"]    = np.sin(2*np.pi*df["hour"]    / 24)
    df["hour_cos"]    = np.cos(2*np.pi*df["hour"]    / 24)
    df["weekday_sin"] = np.sin(2*np.pi*df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2*np.pi*df["weekday"] / 7)
    return df

def _add_lags(df):
    # shift global, igual que en el entrenamiento
    df["lag1"] = df["internet_scaled"].shift(1)
    df["lag2"] = df["internet_scaled"].shift(2)
    return df

def _build_rows_next_interval(df2int):
    """
    Recibe exactamente DOS intervalos de 10' y genera una fila por CellID
    con las características para la siguiente secuencia (t + 10 min).
    """
    df2int.sort_values(["time_bin", "CellID"], inplace=True)
    df2int["internet_scaled"] = scaler.transform(df2int[["internet"]])

    df2int = _add_lags(df2int).dropna(subset=["lag1", "lag2"])

    last_tb   = df2int["time_bin"].max()
    next_tb   = last_tb + pd.Timedelta(minutes=10)
    next_hour = next_tb.hour
    next_wd   = next_tb.weekday()

    # Tomamos únicamente las filas del último instante (t0)
    last_rows = df2int[df2int["time_bin"] == last_tb].copy()

    X = pd.DataFrame({
        "hour_sin"   : np.sin(2*np.pi*next_hour / 24),
        "hour_cos"   : np.cos(2*np.pi*next_hour / 24),
        "weekday_sin": np.sin(2*np.pi*next_wd   / 7),
        "weekday_cos": np.cos(2*np.pi*next_wd   / 7),
        "lag1"       : last_rows["internet_scaled"],
        "lag2"       : last_rows["lag1"],
        "idx"        : last_rows["idx"],
        "CellID"     : last_rows["CellID"],
    })

    # misma orden de columnas que en el entrenamiento
    X = X[["hour_sin","hour_cos","weekday_sin","weekday_cos",
           "lag1","lag2","idx","CellID"]]

    # Categoría con el mismo orden 1…10000
    X["CellID"] = X["CellID"].astype("category")
    X["CellID"] = X["CellID"].cat.set_categories(range(1, 10001))

    return X, next_tb

# ---------------------------------------------------------------
@app.get("/status")
def health():
    return {"status":"ok","msg":"LightGBM predictor ready"}

@app.post("/predict/lightgbm_csv")
async def predict(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(
            pd.io.common.StringIO((await file.read()).decode()),
            dtype={"date":str,"hour":int,"minute":int,
                   "CellID":int,"internet":float,"weekday":int,"idx":int}
        )
        df = _time_bin(df)
        if df["time_bin"].nunique() != 2:
            raise ValueError("Se requieren exactamente 2 intervalos de 10 min.")

        df = _add_features(df)
        X_next, next_tb = _build_rows_next_interval(df)

        t0 = time.perf_counter()
        pred_scaled = model.predict(X_next)
        inf_t = time.perf_counter() - t0

        pred = scaler.inverse_transform(pred_scaled.reshape(-1,1)).flatten()
        total_internet = float(pred.sum())

        return {"status":"success",
                "predicted_total_internet": total_internet,
                "inference_time": inf_t,
                "predicted_for": str(next_tb)}
    except Exception as e:
        logger.error(f"LightGBM error: {e}")
        return {"status":"error","message":str(e)}

if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="192.168.1.39", port=8004)
