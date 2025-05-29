from fastapi import FastAPI, File, UploadFile
import pandas as pd
import requests
import numpy as np
import os
import time
from datetime import datetime
import pulp  # Para la programación lineal.
from typing import Tuple

app = FastAPI()

# URL de los servicios de predicción
PREDICTOR_URLS = {
    "lightgbm_callout": "http://192.168.1.39:8000/predict/lightgbm_csv",
    "lightgbm_smsout": "http://192.168.1.39:8002/predict/lightgbm_csv",
    "lightgbm_internet": "http://192.168.1.39:8004/predict/lightgbm_csv",
    "lstm_callout": "http://192.168.1.39:8001/predict/lstm_csv",
    "lstm_smsout": "http://192.168.1.39:8003/predict/lstm_csv",
    "lstm_internet": "http://192.168.1.39:8005/predict/lstm_csv"
}
AGENT_URL = "http://192.168.1.70:8051"

# Valores mínimos y máximos de las predicciones de a escalas
MIN_MAX_VALUES = {
    "callout": {"min": 971.1443768305097, "max": 92086.40016479186},
    "smsout": {"min": 3108.3292205315656, "max": 48830.186264552496},
    "internet": {"min": 280734.3879388445, "max": 1059772.9383324143}
}

results = {"smape": {}, "totals": {}, "bandwidth": {}}
NUM_ITERATIONS = 7

def send_csv_to_predictor(df: pd.DataFrame, url: str, tmp_csv: str):
    df.to_csv(tmp_csv, index=False)
    with open(tmp_csv, "rb") as f:
        files = {"file": (os.path.basename(tmp_csv), f, "text/csv")}
        resp = requests.post(url, files=files)
    return resp.json()

def calculate_smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_pred - y_true) / denominator
    return 100 * np.mean(diff)

def scale_prediction(pred_value: float, traffic_class: str) -> float:
    min_val = MIN_MAX_VALUES[traffic_class]["min"]
    max_val = MIN_MAX_VALUES[traffic_class]["max"]
    # Escalar a 18 Mbit/s y redondear al entero más cercano
    scaled_value = ((pred_value - min_val) / (max_val - min_val)) * 18
    return round(max(0.0, scaled_value), 2)  # Asegurar que no sea negativo y devolver un entero

def allocate_throughput(call_req: float, sms_req: float, net_req: float) -> Tuple[float, float, float]:
    # Límites inferiores ( Eliminar el valor actual y sustituir por el comentado si se quieren límites inferiores dinámicos para utilizar los valores reales en todas las situaciones )
    call_lb = 0 #1 if 0 < call_req < 1 else 0  # 1 Mbit/s solo si 0 < call_req < 1, 1 si hay o no demanda
    sms_lb = 1 #1 if 0 < sms_req < 1 else 0    # 1 Mbit/s si 0 < sms_req < 1, 0 si no hay demanda
    net_lb = 1 #1 if 0 < net_req < 1 else 0    # 1 Mbit/s si 0 < net_req < 1, 0 si no hay demanda

    # Límites superiores nunca estarán por debajo del límite inferior
    call_ub = min(call_req, 18)  # Asegura que call_ub >= call_lb y no exceda 18 (escalado de tráficos)
    sms_ub = max(sms_req, sms_lb)
    net_ub = max(net_req, net_lb)

    # Problema de optimización
    prob = pulp.LpProblem("Bandwidth_Allocation", pulp.LpMaximize)

    # Variables de decisión con límites ajustados
    x1 = pulp.LpVariable("x1_callout", lowBound=call_lb, upBound=call_ub, cat='Continuous')
    x2 = pulp.LpVariable("x2_smsout", lowBound=sms_lb, upBound=sms_ub, cat='Continuous')
    x3 = pulp.LpVariable("x3_internet", lowBound=net_lb, upBound=net_ub, cat='Continuous')

    # Función objetivo
    prob += 3 * x1 + 2 * x2 + x3, "Utility"

    # Restricción de capacidad. Aquí se garantizan 2 Mbit/s = 20 Mbit/s (capacidad máxima del enlace)- 18 Mbit/s(Limite superior de los tráficos).
    prob += x1 + x2 + x3 <= 20, "Capacity"

    # Resolver el problema
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Obtener los valores óptimos y redondearlos
    call_bw = 1 if 0 < call_req < 1 else round(x1.varValue)
    sms_bw = round(x2.varValue)
    net_bw = round(x3.varValue)

    return call_bw, sms_bw, net_bw


@app.post("/upload_datasets")
async def upload_datasets(
    callout_tmp_lightgbm: UploadFile = File(...), callout_tmp_lstm: UploadFile = File(...),
    smsout_tmp_lightgbm: UploadFile = File(...), smsout_tmp_lstm: UploadFile = File(...),
    internet_tmp_lightgbm: UploadFile = File(...), internet_tmp_lstm: UploadFile = File(...)
):
    print("En espera de los dataset del script de simulación...")
    files = {
        "callout_tmp_lightgbm": callout_tmp_lightgbm, "callout_tmp_lstm": callout_tmp_lstm,
        "smsout_tmp_lightgbm": smsout_tmp_lightgbm, "smsout_tmp_lstm": smsout_tmp_lstm,
        "internet_tmp_lightgbm": internet_tmp_lightgbm, "internet_tmp_lstm": internet_tmp_lstm
    }
    for name, file in files.items():
        with open(f"{name}.csv", "wb") as f:
            f.write(await file.read())
    print("Datasets recibido y guardado")
    return {"status": "Datasets received"}

@app.post("/start_simulation")
def start_simulation(data: dict):
    iteration = data["iteration"]
    totals = data["totals"]
    print(f"Notificación de inicio de simulación recibida para la iteración {iteration}")

    # Paso 3: Solicitar métricas al agente SDN
    print("Solicitud de métricas al agente SDN")
    metrics_response = requests.get(f"{AGENT_URL}/metrics")
    metrics = metrics_response.json()
    print(f"Métricas recibidas: {metrics}")

    # Paso 5: Obtener predicciones (LightGBM)
    predictions = {}
    for cls in ["callout", "smsout", "internet"]:
        model = "lightgbm"
        df = pd.read_csv(f"{cls}_tmp_{model}.csv")
        url = PREDICTOR_URLS[f"{model}_{cls}"]
        resp = send_csv_to_predictor(df, url, f"{cls}_tmp_{model}.csv")
        if resp.get("status") == "success":
            pred_key = f"predicted_total_{cls}"
            predictions[f"{model}_{cls}"] = resp[pred_key]
            predicted_for = datetime.strptime(resp["predicted_for"], "%Y-%m-%d %H:%M:%S")
            inference_time = resp["inference_time"]
            real_value = totals[cls]
            smape = calculate_smape(real_value, predictions[f"{model}_{cls}"])
            print(f"{model.upper()} {cls} - Iteración {iteration}: Obtenidas {len(df)} filas para el rango {df['timestamp'].min()} - {df['timestamp'].max()}")
            print(f"Predicción para {predicted_for}: {predictions[f"{model}_{cls}"]}")
            print(f"Valor real: {real_value}")
            print(f"SMAPE: {smape:.2f}%")
            print(f"Tiempo de inferencia: {inference_time:.4f} seg")
        else:
            print(f"Error en la predicción para {model}_{cls}: {resp.get('message', 'Respuesta inválida')}")

    # Paso 6: Calcular SMAPE (LightGBM)
    for cls in ["callout", "smsout", "internet"]:
        model = "lightgbm"
        if f"{model}_{cls}" in predictions:
            real = totals[cls]
            pred = predictions[f"{model}_{cls}"]
            smape = calculate_smape(real, pred)
            if f"{model}_{cls}" not in results["smape"]:
                results["smape"][f"{model}_{cls}"] = []
            results["smape"][f"{model}_{cls}"].append({"smape": smape, "pred": pred})
        else:
            print(f"No se pudo calcular SMAPE para {model}_{cls} debido a predicción faltante")

    # Paso 7: Asignar ancho de banda mediante predicciones
    model = "lightgbm"
    reqs = {}
    for cls in ["callout", "smsout", "internet"]:
        if f"{model}_{cls}" in predictions:
            pred_value = predictions[f"{model}_{cls}"]
            reqs[cls] = scale_prediction(pred_value, cls)
        else:
            print(f"Asignando requerimiento por defecto para {cls} debido a predicción faltante")
            reqs[cls] = 30  # Valor por defecto razonable

    call_bw, sms_bw, net_bw = allocate_throughput(reqs["callout"], reqs["smsout"], reqs["internet"])
    config = {"callout": call_bw, "smsout": sms_bw, "internet": net_bw}

    # Paso 8: Enviar la configuración al agente SDN
    print(f"Envío de la configuración al agente SDN: {config}")
    apply_response = requests.post(f"{AGENT_URL}/apply_config", json=config)
    if apply_response.status_code == 200:
        print("Configuración aplicada correctamente al OVS")
        results["bandwidth"][iteration] = config
        results["totals"][iteration] = totals

    # Paso 11: Limpieza temporal datasets
    for cls in ["callout", "smsout", "internet"]:
        for model in ["lightgbm", "lstm"]:
            os.remove(f"{cls}_tmp_{model}.csv")
    print("Conjuntos de dataset temporales eliminados")

    if iteration == NUM_ITERATIONS:
        # Restablecer la configuración inicial
        initial_config = {"callout": 10, "smsout": 10, "internet": 10}
        requests.post(f"{AGENT_URL}/apply_config", json=initial_config)
        print("Restablecer la configuración inicial del ancho de banda")

    return {"status": "Simulation step completed"}

if __name__ == "__main__":
    print("Controlador SDN iniciándose, a la espera del inicio de la simulación...")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8052)
