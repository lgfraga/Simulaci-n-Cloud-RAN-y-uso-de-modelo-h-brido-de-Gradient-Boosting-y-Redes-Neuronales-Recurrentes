  # Predicción de tráfico en redes móviles mediante modelo híbrido de Gradient Boosting y Redes Neuronales Recurrentes

## Resumen

Este proyecto se centra en desarrollar un marco comparativo para evaluar dos alternativas de predicción de tráfico móvil utilizando datos históricos de la ciudad de Milán. Se implementa un sistema en el que el modelo **LightGBM** se emplea tanto para la predicción directa como para la ingeniería de características, y se contrasta con una red neuronal **LSTM** diseñada a partir del modelo LightGBM para capturar dependencias temporales a largo plazo. La evaluación se realiza mediante métricas como el **error cuadrático medio (MSE)** y el **error porcentual medio absoluto simétrico (SMAPE)**, con el fin de determinar cuál de los dos modelos ofrece mayor precisión predictiva. Además, se propone integrar el servicio predictor en un entorno simulado de **Cloud-RAN**, utilizando una **API REST** para comunicar el predictor con un controlador **SDN** desplegado en una máquina virtual (vm2), que orquesta los ajustes dinámicos del tráfico en un conmutador virtual (**OVS**) en otra máquina virtual (vm3). La simulación incluye la generación de tráfico agregado ascendente desde una máquina virtual que abstrae a la unidad distribuida (**vDU**), replicando patrones realistas de tráfico móvil mediante dataset y el uso de iPerf3 para generar a los tráficos. Los resultados aportan una visión integral sobre el equilibrio entre precisión y eficiencia en la gestión del tráfico móvil en escenarios de redes virtualizadas.

## Introducción

La evolución de las redes móviles hacia la quinta generación (**5G**) y la consolidación de arquitecturas virtualizadas como **Cloud-RAN** han generado la necesidad imperiosa de gestionar de forma óptima el tráfico de red para garantizar una alta calidad de servicio y evitar la congestión. En este contexto, la predicción precisa del tráfico se convierte en una herramienta fundamental para anticipar picos de demanda y ajustar dinámicamente los recursos disponibles. Este proyecto aborda el desarrollo de un sistema que integra las fortalezas del modelo de aprendizaje automático **LightGBM** y del modelo de aprendizaje profundo **LSTM**, para lograr predicciones con una precisión aceptables en distintos horizontes temporales.

Dado que en un entorno **Cloud-RAN** el tráfico saliente de los usuarios hacia el núcleo de la red **5G** suele generar mayores cuellos de botella, se focaliza el estudio en datos de salida concretos: **llamadas salientes**, **envíos de SMS** y **consumo de datos de Internet**. Estas variables representan el flujo de información que puede saturar los enlaces y afectar directamente la calidad del servicio, haciendo su predicción clave para estrategias de asignación y reconfiguración de recursos en tiempo real.

La metodología se apoya en un enfoque de gestión de proyectos, articulando procesos esenciales como el desarrollo del plan, la definición del alcance mediante **EDT**, y la elaboración del cronograma y estimación de costes (en términos de esfuerzo). Aunque no se dispone de un entorno real de **Cloud-RAN**, se implementa un simulador mediante una infraestructura virtualizada con **VirtualBox**, utilizando tres máquinas virtuales interconectadas que replican un entorno **Cloud-RAN** muy simplificado.

## Cumplimiento de objetivos del proyecto

### Objetivo principal
**Desarrollar y establecer un marco comparativo e integrador para evaluar los modelos de predicción de tráfico móvil basados en LightGBM y LSTM, utilizando datos históricos de tráfico de la ciudad de Milán y determinando cuál ofrece mayor precisión predictiva en un entorno que simula Cloud-RAN.**

- **Cumplimiento**: Se desarrolló un marco comparativo integrando **LightGBM** y **LSTM**, utilizando datos históricos de Milán. La evaluación se realizó con métricas como **MSE** y **SMAPE**, y los modelos se desplegaron en un entorno simulado de **Cloud-RAN** con máquinas virtuales (**vm1**, **vm2**, **vm3**) y un predictor externo en el host. Los resultados, disponibles en los directorios `LightGBM-LSTM` y `Cloud-RAN`, muestran que ambos modelos logran buena precisión con un SMAPE < 10 % en la mayoría de ensayos (Peor rendimiento cuando hay mínimo tráfico).

### Objetivos secundarios

1. **Procesamiento y análisis de datos**: Desarrollar un flujo de trabajo para el preprocesamiento y análisis del conjunto de datos históricos de tráfico móvil de Milán, identificando patrones temporales esenciales para la predicción.
   - **Cumplimiento**: El script `generar_dataset.v03.py` en `LightGBM-LSTM` procesa los datos, generando datasets para **callout**, **smsout** e **internet**.

2. **Implementación del modelo LightGBM**: Desarrollar un sistema basado en **LightGBM** con un **SMAPE** inferior al 10% en valores a escala real, evaluado con **MSE** y coste computacional.
   - **Cumplimiento**: Los scripts `Regresion_LightGBM_callout_lags_v5.py` (y equivalentes para **smsout** e **internet**) en `LightGBM-LSTM` implementan el modelo, logrando un **SMAPE** < 10%, con métricas y tiempos de inferencia.

3. **Desarrollo del modelo LSTM**: Implementar una red **LSTM** usando **LightGBM** para ingeniería de características, buscando un **SMAPE** entre 10% y 20%, o inferior al 10% si es posible.
   - **Cumplimiento**: `Integración_LightGBM+LSTM_predictor_callout_v9_128dim.py` (y equivalentes) en `LightGBM-LSTM` desarrolla el modelo **LSTM**, alcanzando un **SMAPE** muy adecuado, en pruebas < 10%, gracias a características cíclicas y lags.

4. **Comparativa de modelos**: Evaluar los modelos con **SMAPE**, **MSE** y **R²** como complemento, y comparando tiempos de inferencia.
   - **Cumplimiento**: Los scripts `evaluar_lightgbm_largo_horizonte.py`, `evaluar_lstm_largo_horizonte.py` y `comparar_largo_horizonte.py` en `LightGBM-LSTM` generan las comparativas que se detallan en la memoria del proyecto.
5. **Diseño modular y despliegue en un entorno simulado de Cloud-RAN**: Diseñar un sistema con tres VMs y servicios predictores en el host, comunicados vía **API REST**.
   - **Cumplimiento**: El directorio `Cloud-RAN` contiene los scripts de inicialización (`configurar_vlan_vm1.sh`, `inicializar_red.sh`), simulación (`servicio_sim_traf.py`), controlador **SDN** (`controlador_sdn_lightgbm.py`, `controlador_sdn_lstm.py`) y agente **SDN** (`agente_sdn.py`), integrados con predictores en `LightGBM-LSTM` (`servidor_LightGBM_callout_v09.py`, `servidor_LightGBM_smsout_v09.py`,`servidor_LightGBM_internet_v09.py`,`servidor_LSTM_callout_v09.py`,`servidor_LSTM_smsout_v09.py`,`servidor_LSTM_internet_v09.py`).

## Estructura del repositorio

- **`Cloud-RAN`**: Ficheros de inicialización de red para **vm1** y **vm3**, entornos virtuales y scripts para simulación (**vm1**), controlador **SDN** (**vm2**) y agente **SDN** (**vm3**). Materializa el objetivo 5.
- **`LightGBM-LSTM`**: Scripts y ficheros para procesamiento de datos, implementación y comparación de modelos **LightGBM** y **LSTM**. Cumple los objetivos 1-4.
- **`log`**: Logs de **iPerf3** con tráficos generados en simulaciones.
- **`vm's_Virtualbox`**: Máquinas virtuales utilizadas en la simulación que se han comprimido y dividido en varias partes con extensión .zip, listas para descargar y utilizar.

### Descripción de ficheros

#### `LightGBM-LSTM`
- **`dataset_callout.csv`, `dataset_smsout.csv`, `dataset_internet.csv`**: Datasets procesados de tráfico móvil.
- **`LightGBM_callout_001.pkl`, etc.**: Modelos entrenados de **LightGBM**.
- **`LSTM_callout_001.pth`, etc.**: Modelos entrenados de **LSTM**.

#### `Cloud-RAN`
- **`configurar_vlan_vm1.sh`**: Configura VLANs en **vm1**.
- **`inicializar_red.sh`**: Inicializa **OVS** y VLANs en **vm3**.
- **`servicio_sim_traf.py`**: Genera tráfico simulado en **vm1**.
- **`controlador_sdn_lightgbm.py`, `controlador_sdn_lstm.py`**: Controladores **SDN** en **vm2** para los servicios de predicción LightGBM y LSTM según corresponda.
- **`agente_sdn.py`**: Agente **SDN** en **vm3**, encargado de aplicar la configuración resultante en función de la predicción obtenida del los servicios predictores.

## Guía de usuario para replicar la simulación

### Requisitos
- **Hardware**: Estación de trabajo con Windows 11, procesador Intel i9 (16 núcleos), 64 GB RAM, NVIDIA RTX 4090 (24 GB VRAM) como recomendado (Minimo 32 GB RAM).
- **Software**: Python 3.13, VirtualBox, Ubuntu Server 24.04.2 LTS, bibliotecas: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `lightgbm`, `torch`, `fastapi`, `uvicorn`, `requests`.

### Opción 1: Usando las VMs del repositorio

1. **Inicializar vm3 (6 núcleos y 8 GB RAM)**
   - Iniciar sesión: `User: administrador`, `Passwd: tfm2025`
   - Iniciar superusuario: `sudo su`
   - Ejecutar: `./inicializar_red.sh`
   - Acceder e iniciar entorno: `cd /home/administrador/agenteSDN`, `source agenteSDN/bin/activate`
   - Ejecutar agente SDN: `python agente_sdn.py`
   - Monitorizar: `http://192.168.1.71:3000` (Grafana).

2. **Inicializar vm2 (2 núcleos y 2 GB RAM)**
   - Iniciar sesión: `User: administrador`, `Passwd: tfm2025`
   - Iniciar superusuario: `sudo su`
   - Acceder e iniciar entorno: `cd /home/administrador/C_SDN`, `source C_SDN/bin/activate`
   - Iniciar controlador SDN:
     - LightGBM: `python controlador_sdn_lightgbm.py`
     - LSTM: `python controlador_sdn_lstm.py`

3. **Iniciar servicios predictores**
   - Desde `LightGBM-LSTM` en el host:
     - LightGBM: `python servidor_LightGBM_callout_v09.py`, `python servidor_LightGBM_smsout_v09.py`, `python servidor_LightGBM_internet_v09.py`
     - LSTM: `python servidor_LSTM_callout_v09.py`, `python servidor_LSTM_smsout_v09.py`, `python servidor_LSTM_internet_v09.py`

4. **Inicializar vm1 (4 núcleos y 8 GB RAM)**
   - Iniciar sesión: `User: administrador`, `Passwd: tfm2025`
   - Iniciar superusuario: `sudo su`
   - Ejecutar: `./configurar_vlan_vm1.sh`
   - Acceder e iniciar entorno: `cd /home/administrador/simulacion_trafico/vDU`, `source vDU/bin/activate`
   - Iniciar simulación: `python servicio_sim_traf.py`
   - Nota: Ajustar `initial_start_dt` en `servicio_sim_traf.py` para la fecha/hora de inicio. 7 iteraciones = 1 hora.

### Opción 2: Clonar los directorios de forma independiente en cada una de las vm's que haya creado

1. **Preparar VMs**
   - En cada VM (**vm1**, **vm2**, **vm3**):
     - Iniciar como superusuario: `sudo su`
     - Instalar Python 3.13: `sudo add-apt-repository ppa:deadsnakes/ppa`, `sudo apt install python3.13-full`
     - Instalar dependencias: `sudo apt install -y python3-pip nodejs npm`

2. **Configurar Entornos**
   - **vm1**: `cd /home/administrador/simulacion_trafico/vDU`, `python3.13 -m venv vDU`, `source vDU/bin/activate`, `pip install fastapi uvicorn requests`
   - **vm2**: `cd /home/administrador/C_SDN`, `python3.13 -m venv C_SDN`, `source C_SDN/bin/activate`, `pip install fastapi uvicorn requests`
   - **vm3**: `cd /home/administrador/agenteSDN`, `python3.13 -m venv agenteSDN`, `source agenteSDN/bin/activate`, `pip install fastapi uvicorn requests`

3. **Ejecutar pasos de la Opción 1**
   - Seguir los pasos 1-4 de la opción 1.

### Monitorización con Prometheus y Grafana
Ver guía detallada en: "Prometheus + Grafana.txt".
- **Acceso**: `http://192.168.1.71:3000`, `User: admin`, `Passwd: tfm2025`

### Simulación y resultados
- **Ver el video: Simulación C-RAN utilizando servicios predictores LightGBM. Domingo 2013/11/10 12H(Fin de semana)**:
[![Ver el video: Simulación C-RAN utilizando servicios predictores LightGBM. Domingo 2013/11/10 12H](https://img.youtube.com/vi/34ztSiY5ZBA/hqdefault.jpg)](https://www.youtube.com/watch?v=34ztSiY5ZBA)


- **Ver el video: Simulación C-RAN utilizando servicios predictores LSTM. Domingo 2013/11/10 12H - 13H (Fin de semana)**:
[![Ver el video: Simulación C-RAN utilizando servicios predictores LightGBM. Domingo 2013/11/10 12H](https://img.youtube.com/vi/TeoQ7JCzCG8/hqdefault.jpg)](https://www.youtube.com/watch?v=TeoQ7JCzCG8)

El proyecto logró desarrollar un marco comparativo, demostrando que el modelo híbrido **Gradient Boosting** y **Redes Neuronales Recurrentes** ofrecen una excelente precisión predictiva en un entorno simulado de **Cloud-RAN**. Los resultados, soportados por una simulación a escala, aportan valor para estrategias de gestión de tráfico móvil en redes **5G** virtualizadas.
