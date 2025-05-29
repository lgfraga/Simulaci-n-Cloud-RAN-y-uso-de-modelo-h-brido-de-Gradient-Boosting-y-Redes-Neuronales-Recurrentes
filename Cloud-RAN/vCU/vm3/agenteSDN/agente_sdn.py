from fastapi import FastAPI, HTTPException
import subprocess
import json
import os
import time  # Para esperar 30 segundos
import re   # Para expresiones regulares en la búsqueda dentro de los log

app = FastAPI()

# Directorio donde se encuentran los logs de iperf3
LOG_DIR = "/home/administrador/agenteSDN"

def run_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode().strip()
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error executing command: {e.stderr.decode()}")

def obtener_bitrate_ultimo_intervalo(log_filepath):
    """
    Extrae los valores de bitrate de las últimas 15 líneas del log de iperf3,
    calcula su promedio y lo retorna en formato "X Mbits/sec".

    Args:
        log_filepath (str): Ruta completa al fichero log.
    
    Returns:
        str: El promedio del bitrate en formato "X Mbits/sec", o None si no se encuentran datos.
    """
    # Patrón para capturar tanto Mbits/sec como Kbits/sec
    pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(Mbits/sec|Kbits/sec)')
    bitrates = []

    if not os.path.exists(log_filepath):
        print(f"Error: El archivo {log_filepath} no se encontró.")
        return None

    try:
        with open(log_filepath, 'r') as file:
            lines = file.readlines()
            # Tomar las últimas 15 líneas (o todas si hay menos)
            last_lines = lines[-15:]
            for line in last_lines:
                match = pattern.search(line)
                if match:
                    value = float(match.group(1))
                    unit = match.group(2)
                    # Si la unidad es Kbits/sec, se convierte a Mbits/sec
                    if unit == 'Kbits/sec':
                        value /= 1000
                    bitrates.append(value)
    except Exception as e:
        print(f"Error al leer el archivo {log_filepath}: {e}")
        return None

    if bitrates:
        promedio_bitrate = sum(bitrates) / len(bitrates)
        return f"{promedio_bitrate:.2f} Mbits/sec"
    else:
        return None

@app.get("/status")
def check_status():
    print("Estado del agente SDN solicitado...")
    return {"status": "OK"}

@app.get("/metrics")
def get_metrics():
    print("Métricas solicitadas desde el Controlador SDN")
    print("Esperando 30 segundos hasta que se estabilice el tráfico...")
    
    # Esperar 30 segundos para que el tráfico se estabilice
    time.sleep(30)
    print("Procediendo a la obtención de métricas...")
    
    # Obtener las tasas configuradas desde OVS para los puertos internos
    call_rate = float(run_command("ovs-vsctl get interface callout ingress_policing_rate")) / 1000
    sms_rate = float(run_command("ovs-vsctl get interface smsout ingress_policing_rate")) / 1000
    net_rate = float(run_command("ovs-vsctl get interface internet ingress_policing_rate")) / 1000
    
    # Obtener el throughput real desde los logs de iperf3
    call_act = obtener_bitrate_ultimo_intervalo(os.path.join(LOG_DIR, "iperf_servidor_callout.log"))
    sms_act = obtener_bitrate_ultimo_intervalo(os.path.join(LOG_DIR, "iperf_servidor_smsout.log"))
    net_act = obtener_bitrate_ultimo_intervalo(os.path.join(LOG_DIR, "iperf_servidor_internet.log"))
    
    # Si no se encuentra un valor, asignar 0.0 Mbits/sec por defecto
    call_act = call_act if call_act else "0.0 Mbits/sec"
    sms_act = sms_act if sms_act else "0.0 Mbits/sec"
    net_act = net_act if net_act else "0.0 Mbits/sec"
    
    metrics = {
        "callout_rate": call_rate, "smsout_rate": sms_rate, "internet_rate": net_rate,
        "call_act": call_act, "sms_act": sms_act, "net_act": net_act
    }
    print(f"Métricas recuperadas: {metrics}")
    return metrics

@app.post("/apply_config")
def apply_config(config: dict):
    print(f"Nueva configuración recibida: {config}")
    call_bw = config.get("callout")
    sms_bw = config.get("smsout")
    net_bw = config.get("internet")

    # Aplicar configuraciones a OVS con sudo
    run_command(f"sudo ovs-vsctl set interface callout ingress_policing_rate={call_bw * 1000}")
    run_command(f"sudo ovs-vsctl set interface callout ingress_policing_burst={call_bw * 100}")
    run_command(f"sudo ovs-vsctl set interface smsout ingress_policing_rate={sms_bw * 1000}")
    run_command(f"sudo ovs-vsctl set interface smsout ingress_policing_burst={sms_bw * 100}")
    run_command(f"sudo ovs-vsctl set interface internet ingress_policing_rate={net_bw * 1000}")
    run_command(f"sudo ovs-vsctl set interface internet ingress_policing_burst={net_bw * 100}")

    # Verificar la configuración aplicada
    metrics = get_metrics()
    print(f"Configuración aplicada y verificada: {metrics}")
    return {"status": "Configuration applied", "metrics": metrics}

if __name__ == "__main__":
    print("Agente SDN iniciándose...")
    print("A la espera de peticiones del Controlador SDN...")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8051)