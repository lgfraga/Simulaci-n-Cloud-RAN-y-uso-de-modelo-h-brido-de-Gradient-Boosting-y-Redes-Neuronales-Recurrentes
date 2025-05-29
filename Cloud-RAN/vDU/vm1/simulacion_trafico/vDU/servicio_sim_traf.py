import pandas as pd
import numpy as np
import subprocess
import requests
from datetime import datetime, timedelta
import os
import time
from typing import Tuple
import shutil  # Añadido para comprobar el ejecutable iPerf3

# Constants
IP_CALLOUT = "10.110.0.1"  # IP vLAN para callout
IP_SMSOUT = "10.112.0.1"   # IP vLAN para smsout
IP_INTERNET = "10.113.0.1" # IP vLAN para internet
IP_ENP0S8 = "10.110.0.2"   # IP vLAN
IP_ENP0S9 = "10.112.0.2"   # IP vLAN
IP_ENP0S10 = "10.113.0.2"  # IP vLAN
PORTS = {"callout": 9000, "smsout": 9001, "internet": 9002}
DATASETS = ["dataset_callout.csv", "dataset_smsout.csv", "dataset_internet.csv"]
TMP_FILES = {
    "callout": {"lightgbm": "callout_tmp_lightgbm.csv", "lstm": "callout_tmp_lstm.csv"},
    "smsout": {"lightgbm": "smsout_tmp_lightgbm.csv", "lstm": "smsout_tmp_lstm.csv"},
    "internet": {"lightgbm": "internet_tmp_lightgbm.csv", "lstm": "internet_tmp_lstm.csv"}
}
CONTROLLER_URL = "http://192.168.1.60:8052/upload_datasets"
AGENT_URL = "http://192.168.1.70:8051/status"
START_URL = "http://192.168.1.60:8052/start_simulation"
SIM_DURATION = 600  # 10 minutos en segundos
NUM_ITERATIONS = 7 # 7 = 1H de ensayo. Ajustar aquí la duración de la simulación.

# Directorio donde se guardarán los logs de iperf3
LOG_DIR = "/home/administrador/simulacion_trafico/vDU"

# Valores mínimos y máximos de la escala
MIN_MAX_VALUES = {
    "callout": {"min": 971.1443768305097, "max": 92086.40016479186},
    "smsout": {"min": 3108.3292205315656, "max": 48830.186264552496},
    "internet": {"min": 280734.3879388445, "max": 1059772.9383324143}
}

def load_dataset(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv, dtype={
        "date": str, "hour": int, "minute": int, "CellID": int,
        "callout": float, "weekday": int, "idx": int
    }).rename(columns={"callout": path_csv.split("_")[1].split(".")[0]})
    df["timestamp"] = pd.to_datetime(
        df["date"] + " " + df["hour"].astype(str).str.zfill(2) + ":" + df["minute"].astype(str).str.zfill(2),
        format="%Y-%m-%d %H:%M"
    )
    df.sort_values(by=["timestamp", "CellID"], inplace=True, ignore_index=True)
    return df

def filter_dataset(df: pd.DataFrame, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    mask = (df["timestamp"] >= start_dt) & (df["timestamp"] < end_dt)
    return df.loc[mask].copy()

def create_tmp_datasets(df: pd.DataFrame, T: datetime, model_type: str, traffic_class: str) -> pd.DataFrame:
    if model_type == "lightgbm":
        start_dt = T - timedelta(minutes=20)  # Últimos 20 minutos antes de T
        end_dt = T
    elif model_type == "lstm":
        start_dt = T - timedelta(hours=4)  # Últimas 4 horas antes de T
        end_dt = T
    else:
        raise ValueError("Tipo de modelo no compatible")
    return filter_dataset(df, start_dt, end_dt)

def calculate_total_traffic(df: pd.DataFrame, timestamp: datetime, traffic_class: str) -> float:
    # Calcular el total de la secuencia específica en timestamp
    df_seq = df[df["timestamp"] == timestamp]
    return df_seq[traffic_class].sum()

def scale_traffic(total_traffic: float, traffic_class: str) -> float:
    min_val = MIN_MAX_VALUES[traffic_class]["min"]
    max_val = MIN_MAX_VALUES[traffic_class]["max"]
    # Escalar a 18 Mbit/s y redondear al más cercano (Capacidad máxima del canal 20 Mbit/s)
    scaled_traffic = ((total_traffic - min_val) / (max_val - min_val)) * 18
    return round(max(0.0, scaled_traffic), 2)  # Asegurar no negativo y devolver flotante con dos decimales

def allocate_throughput(call_req: float, sms_req: float, net_req: float) -> Tuple[float, float, float]:
    # Asignar directamente los valores escalados
    call_bw = call_req
    sms_bw = sms_req
    net_bw = net_req
    return call_bw, sms_bw, net_bw

def generate_traffic(call_bw: float, sms_bw: float, net_bw: float, totals: dict, iteration: int):
    """
    Lanza tres procesos iperf3 en paralelo (uno por clase de tráfico),
    envía la notificación al Controlador SDN justo después de iniciarlos y espera a que terminen.
    """
    # Comprobar que iperf3 está instalado
    if not shutil.which("iperf3"):
        raise FileNotFoundError(
            "iperf3 no está instalado o no está en PATH. "
            "Instálalo con sudo apt install iperf3."
        )

    processes = []

    # Proceso para callout
    processes.append(
        subprocess.Popen(
            [
                "iperf3",
                "-c", IP_CALLOUT,
                "-p", str(PORTS["callout"]),
                "-B", IP_ENP0S8,
                "-u",
                "-b", f"{call_bw:.2f}M",
                "-t", str(SIM_DURATION),
                "--logfile", os.path.join(LOG_DIR, "iperf_cliente_callout.log"),
            ]
        )
    )

    # Proceso para smsout
    processes.append(
        subprocess.Popen(
            [
                "iperf3",
                "-c", IP_SMSOUT,
                "-p", str(PORTS["smsout"]),
                "-B", IP_ENP0S9,
                "-u",
                "-b", f"{sms_bw:.2f}M",
                "-t", str(SIM_DURATION),
                "--logfile", os.path.join(LOG_DIR, "iperf_cliente_smsout.log"),
            ]
        )
    )

    # Proceso para internet
    processes.append(
        subprocess.Popen(
            [
                "iperf3",
                "-c", IP_INTERNET,
                "-p", str(PORTS["internet"]),
                "-B", IP_ENP0S10,
                "-u",
                "-b", f"{net_bw:.2f}M",
                "-t", str(SIM_DURATION),
                "--logfile", os.path.join(LOG_DIR, "iperf_cliente_internet.log"),
            ]
        )
    )

    # Enviar notificación al controlador SDN justo después de iniciar los procesos iPerf3
    requests.post(START_URL, json={"iteration": iteration, "totals": totals})
    print("Notificación al Controlador SDN del inicio de la simulación...")

    # Se espera a que los tres procesos de iPerf3 terminen
    for proc in processes:
        proc.wait()

def main():
    # Cargar conjuntos de dataset completos
    dfs = {dataset.split("_")[1].split(".")[0]: load_dataset(dataset) for dataset in DATASETS}
    initial_start_dt = datetime(2013, 11, 13, 11, 0, 0)  # Inicio de la simulación (Cambiar aquí la fecha y hora para iniciar el ensayo)
    step = timedelta(minutes=10)

    for iteration in range(NUM_ITERATIONS):
        print(f"Inicio de la iteración {iteration + 1}/{NUM_ITERATIONS}")
        T = initial_start_dt + step * iteration

        # Paso 1: Crear datasets temporales
        for traffic_class in ["callout", "smsout", "internet"]:
            df = dfs[traffic_class]
            for model in ["lightgbm", "lstm"]:
                df_tmp = create_tmp_datasets(df, T, model, traffic_class)
                df_tmp.to_csv(TMP_FILES[traffic_class][model], index=False)
        print("Conjuntos de datasets creados correctamente")

        # Paso 2: Enviar los datasets al controlador SDN
        files = {f"{cls}_tmp_{model}": open(TMP_FILES[cls][model], "rb") for cls in TMP_FILES for model in TMP_FILES[cls]}
        response = requests.post(CONTROLLER_URL, files=files)
        for f in files.values():
            f.close()
        print("Conjuntos de datasets enviados al controlador SDN" if response.status_code == 200 else "Error enviando los datasets")

        # Paso 3: Calcular totals para el cálculo del tráfico actual (Tráfico en T)
        if iteration == 0:
            # Para la primera iteración, se utiliza la secuencia anterior a initial_start_dt,
            # que corresponde al total actual (T), y así poder estimar los próximos 10 minutos (T+1) 
            prev_T = initial_start_dt - step
        else:
            prev_T = T - step
        totals = {cls: calculate_total_traffic(dfs[cls], prev_T, cls) for cls in ["callout", "smsout", "internet"]}
        reqs = {cls: scale_traffic(totals[cls], cls) for cls in totals}
        call_bw, sms_bw, net_bw = allocate_throughput(reqs["callout"], reqs["smsout"], reqs["internet"])

        # Paso 4: Generar tráfico y notificar al controlador SDN
        generate_traffic(call_bw, sms_bw, net_bw, totals, iteration + 1)
        print(f"Tráfico generado: callout={call_bw}Mbits/s, smsout={sms_bw}Mbits/s, internet={net_bw}Mbits/s")

        # Paso 5: Limpieza temporal de datasets
        for tmp_files in TMP_FILES.values():
            for file in tmp_files.values():
                if os.path.exists(file):
                    os.remove(file)
        print("Conjuntos de datasets temporales eliminados")

    print("Simulación finalizada")

if __name__ == "__main__":
    main()
