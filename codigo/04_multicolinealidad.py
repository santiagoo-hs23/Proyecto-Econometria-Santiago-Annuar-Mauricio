
# PROYECTO FINAL DE ECONOMETRIA
# ANALISIS DE MULTICOLINEALIDAD

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path

# Localizar el proyecto
carpeta = Path(__file__).resolve().parent.parent

# Cargar datos
ruta = carpeta / "datos" / "processed" / "rendimientos_semanales.csv"
datos = pd.read_csv(ruta)

# Variables explicativas
X = datos[["r_NASDAQ", "r_SOX"]]
X_const = sm.add_constant(X)

# Correlacion entre variables explicativas
print("\nCORRELACION ENTRE NASDAQ Y SOX")
print(X.corr())

# Calcular VIF
tabla_vif = pd.DataFrame({
    "Variable": X.columns,
    "VIF": [
        variance_inflation_factor(X_const.values, i)
        for i in range(1, X_const.shape[1])
    ]
})

print("\nFACTORES DE INFLACION DE LA VARIANZA")
print(tabla_vif.to_string(index=False))

# Guardar resultados
tabla_vif.to_csv(
    carpeta / "datos" / "processed" / "tabla_vif.csv",
    index=False
)
