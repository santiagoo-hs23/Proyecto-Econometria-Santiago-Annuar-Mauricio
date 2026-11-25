
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# 1. Localizar las carpetas del proyecto
carpeta = Path(__file__).resolve().parent.parent
ruta_datos = carpeta / "datos" / "processed" / "rendimientos_semanales.csv"
carpeta_graficas = carpeta / "graficas"
carpeta_graficas.mkdir(parents=True, exist_ok=True)

# 2. Cargar los datos
datos = pd.read_csv(ruta_datos)

# 3. Estimar la regresión múltiple
Y = datos["r_NVDA"]
X = datos[["r_NASDAQ", "r_SOX"]]
X = sm.add_constant(X)

modelo = sm.OLS(Y, X).fit()

# Obtener residuos y valores ajustados
residuos = modelo.resid
ajustados = modelo.fittedvalues

sns.set_theme(style="whitegrid")

# GRÁFICA 1: Residuos vs valores ajustados
plt.figure(figsize=(9, 5))
sns.scatterplot(x=ajustados, y=residuos)
plt.axhline(0, color="red", linestyle="--")
plt.title("Residuos vs valores ajustados")
plt.xlabel("Rendimiento estimado de NVIDIA (%)")
plt.ylabel("Residuo (puntos porcentuales)")
plt.tight_layout()
plt.savefig(carpeta_graficas / "01_residuos_vs_ajustados.png", dpi=300)
plt.close()

# GRÁFICA 2: Histograma de residuos
plt.figure(figsize=(9, 5))
sns.histplot(residuos, bins=25, kde=True)
plt.title("Distribución de los residuos")
plt.xlabel("Residuo (puntos porcentuales)")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig(carpeta_graficas / "02_histograma_residuos.png", dpi=300)
plt.close()

# GRÁFICA 3: Q-Q Plot
fig = sm.qqplot(residuos, line="45", fit=True)
fig.set_size_inches(7, 6)
plt.title("Q-Q Plot de los residuos")
plt.tight_layout()
plt.savefig(carpeta_graficas / "03_qqplot_residuos.png", dpi=300)
plt.close()

# GRÁFICA 4: Residuos a lo largo del tiempo
fechas = pd.to_datetime(datos["fecha"])

plt.figure(figsize=(10, 5))
plt.plot(fechas, residuos)
plt.axhline(0, color="red", linestyle="--")
plt.title("Evolución temporal de los residuos")
plt.xlabel("Fecha")
plt.ylabel("Residuo (puntos porcentuales)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(carpeta_graficas / "04_residuos_tiempo.png", dpi=300)
plt.close()

print("Se generaron correctamente las cuatro gráficas.")
print("Ubicación:", carpeta_graficas)

