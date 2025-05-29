import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
import datetime

# Configuración de la figura
fig_width_pt = 345  
inches_per_pt = 1.0 / 72.27  
golden_mean = (sqrt(5) - 1.0) / 2.0  
fig_width = fig_width_pt * inches_per_pt  
fig_height = fig_width * golden_mean  
fig_size = [fig_width, fig_height]

sns.set_style("ticks")
sns.set_context("paper")

# Definir rango de fechas (filtrando exactamente los 14 días deseados)
fecha_inicio = datetime.datetime(2013, 11, 1, 0, 0, 0)
fecha_fin = datetime.datetime(2013, 11, 14, 23, 59, 59)

# Generar lista de ficheros con el patrón: sms-call-internet-mi-2013-MM-DD.txt
archivos = []
fecha_actual = fecha_inicio.date()
while fecha_actual <= fecha_fin.date():
    archivo = f'dataset/sms-call-internet-mi-2013-{fecha_actual.month:02d}-{fecha_actual.day:02d}.txt'
    archivos.append(archivo)
    fecha_actual += datetime.timedelta(days=1)

# Lista de parámetros a procesar
parametros = ['smsin', 'smsout', 'callin', 'callout', 'internet']

# Diccionario para almacenar cada dataset
datasets = {}

# Procesar cada parámetro
for parametro in parametros:
    df_list = []
    for archivo in archivos:
        try:
            # Forzamos la lectura de hasta 8 columnas, asignándoles nombres fijos
            df = pd.read_csv(
                archivo,
                sep='\t',
                encoding="utf-8-sig",
                header=None,
                names=['CellID', 'datetime', 'countrycode', 'smsin', 'smsout', 'callin', 'callout', 'internet']
            )
        except Exception as e:
            print(f"Error al leer {archivo}: {e}")
            continue
        
        # Convertir la columna 'datetime' (en milisegundos) a objeto datetime
        df['datetime'] = pd.to_numeric(df['datetime'], errors='coerce')
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms', errors='coerce')
        
        # Filtrar registros: mantener solo los datos entre fecha_inicio y fecha_fin
        df = df[(df['datetime'] >= fecha_inicio) & (df['datetime'] <= fecha_fin)]
        if df.empty:
            continue
        
        # Convertir la columna del parámetro a numérico y rellenar valores faltantes con 0
        df[parametro] = pd.to_numeric(df[parametro], errors='coerce').fillna(0)
        
        # Establecer 'datetime' como índice
        df.set_index('datetime', inplace=True)
        
        # Redondear (floor) el índice al intervalo de 10 minutos
        df['time_bin'] = df.index.floor('10min')
        
        # Agrupar por 'time_bin' y 'CellID' y sumar el valor del parámetro
        df_grouped = df.groupby(['time_bin', 'CellID'], as_index=False).sum()
        
        # Crear columnas adicionales para orden: fecha, hora y minuto
        df_grouped['date'] = df_grouped['time_bin'].dt.date
        df_grouped['hour'] = df_grouped['time_bin'].dt.hour
        df_grouped['minute'] = df_grouped['time_bin'].dt.minute
        
        # Día de la semana (0=lunes, 6=domingo)
        df_grouped['weekday'] = df_grouped['time_bin'].dt.weekday
        
        # Índice calculado: (hora + weekday * 24)
        df_grouped['idx'] = df_grouped['time_bin'].dt.hour + (df_grouped['time_bin'].dt.weekday * 24)
        
        # Reordenar columnas:
        # ['date', 'hour', 'minute', 'CellID', <parámetro>, 'weekday', 'idx']
        columns_order = ['date', 'hour', 'minute', 'CellID', parametro, 'weekday', 'idx']
        df_grouped = df_grouped[columns_order]
        
        # Ordenar el dataset por fecha, hora, minuto y CellID
        df_grouped.sort_values(by=['date', 'hour', 'minute', 'CellID'], inplace=True)
        df_list.append(df_grouped)
    
    if df_list:
        # Concatenar los DataFrames de cada fichero para el parámetro actual
        df_concat = pd.concat(df_list, ignore_index=True)
        datasets[parametro] = df_concat.copy()
        # Exportar a CSV ya ordenado
        df_concat.to_csv(f'dataset_{parametro}.csv', index=False)
        
        # Preparar datos para la gráfica:
        # Reconstruir 'time_bin' a partir de 'date', 'hour' y 'minute'
        df_concat['time_bin'] = (pd.to_datetime(df_concat['date'].astype(str))
                                 + pd.to_timedelta(df_concat['hour'], unit='h')
                                 + pd.to_timedelta(df_concat['minute'], unit='m'))
        df_plot = df_concat.groupby('time_bin', as_index=False)[parametro].sum()
        y_values = df_plot[parametro].values
        
        # Crear una figura con 2 subplots (verticales)
        fig, axs = plt.subplots(2, 1, figsize=(fig_width, fig_height*2))

        # Subfigura 1: Gráfica completa de los 2016 intervalos (14 días)
        axs[0].plot(y_values, label=parametro, linewidth=0.2)
        axs[0].set_xticks(np.arange(0, len(y_values), step=144))  # xticks cada día
        axs[0].tick_params(axis='x', labelsize=8, rotation=45)
        axs[0].tick_params(axis='y', labelsize=8)
        axs[0].legend(loc='upper right', fontsize=8)
        sns.despine(ax=axs[0])
        axs[0].set_xlabel("Intervalos de 10 minutos (2016 en 14 días)", fontsize=10)
        axs[0].set_ylabel(f"Total de {parametro} en cada intervalo", fontsize=10)
        axs[0].set_title(f"Comportamiento de {parametro} (agregado cada 10 minutos)", fontsize=12)

        # Subfigura 2: Zoom en un intervalo, por ejemplo del índice 0 al 144 (1 día)
        zoom_start = 0
        zoom_end = 144  # 144 puntos = 24 horas (cada 10 min)
        axs[1].plot(y_values[zoom_start:zoom_end], label=parametro, linewidth=0.4, marker='.', markersize=1)
        axs[1].set_xticks(np.arange(0, zoom_end - zoom_start, step=6))  # xticks cada 6 intervalos (1 hora)
        axs[1].tick_params(axis='x', labelsize=8, rotation=45)
        axs[1].tick_params(axis='y', labelsize=8)
        axs[1].legend(loc='upper right', fontsize=8)
        sns.despine(ax=axs[1])
        axs[1].set_xlabel("Intervalos de 10 minutos (Zoom: 1 día)", fontsize=10)
        axs[1].set_ylabel(f"Total de {parametro}", fontsize=10)
        axs[1].set_title("Zoom en 24 horas", fontsize=12)

        plt.tight_layout()
        plt.savefig(f'Comportamiento_{parametro}.pdf', format='pdf', dpi=330, bbox_inches='tight')
        plt.close()
        print(f"Figura Comportamiento_{parametro}.pdf creada.")
        print("Generando la siguiente ...")
    else:
        print(f"No se han obtenido datos para el parámetro {parametro}.")

# Mostrar las primeras 5 filas de cada dataset generado
for parametro, df in datasets.items():
    print(f"Dataset generado para {parametro}:")
    print(df.head())
    print("\n . \n . \n . \n")
