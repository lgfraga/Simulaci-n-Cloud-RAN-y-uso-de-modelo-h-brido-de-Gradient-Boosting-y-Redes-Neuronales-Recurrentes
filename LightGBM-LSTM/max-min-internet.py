import pandas as pd

# Cargar el dataset
df = pd.read_csv("dataset_internet.csv")

# Convertir las columnas 'date', 'hour' y 'minute' en un objeto datetime.
df["datetime"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"], unit="h") + pd.to_timedelta(df["minute"], unit="m")

# Redondear cada timestamp hacia abajo al intervalo de 10 minutos más cercano.
df["time_bin"] = df["datetime"].dt.floor("10T")

# Agrupar por 'time_bin' y sumar el campo "internet" en cada intervalo de 10 minutos.
internet_sum = df.groupby("time_bin")["internet"].sum()

# Obtener el valor máximo y mínimo, y sus respectivas marcas de tiempo.
max_internet = internet_sum.max()
min_internet = internet_sum.min()

time_of_max = internet_sum.idxmax()  # Marca de tiempo donde se alcanza el valor máximo.
time_of_min = internet_sum.idxmin()  # Marca de tiempo donde se alcanza el valor mínimo.

# Mostrar los resultados, formateando la marca de tiempo al formato deseado.
print("Valor máximo del total de internet en intervalos de 10 minutos:", max_internet)
print("Se produjo en:", time_of_max.strftime("%Y-%m-%d-%H-%M"))

print("\nValor mínimo del total de internet en intervalos de 10 minutos:", min_internet)
print("Se produjo en:", time_of_min.strftime("%Y-%m-%d-%H-%M"))
