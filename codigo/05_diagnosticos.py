
# PROYECTO FINAL DE ECONOMETRIA
# DIAGNOSTICOS DEL MODELO MULTIPLE

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
from statsmodels.stats.stattools import jarque_bera, durbin_watson
from pathlib import Path

carpeta = Path(__file__).resolve().parent.parent
ruta = carpeta / "datos" / "processed" / "rendimientos_semanales.csv"

datos = pd.read_csv(ruta)

Y = datos["r_NVDA"]
X = sm.add_constant(datos[["r_NASDAQ", "r_SOX"]])

modelo = sm.OLS(Y, X).fit()
residuos = modelo.resid

print("\nDIAGNOSTICOS DEL MODELO MULTIPLE")

# 1. Heterocedasticidad
bp = het_breuschpagan(residuos, X)
print("\n1. BREUSCH-PAGAN")
print("Estadistico LM:", round(bp[0], 4))
print("p-valor:", round(bp[1], 6))

# 2. Autocorrelacion
dw = durbin_watson(residuos)
bg = acorr_breusch_godfrey(modelo, nlags=4)

print("\n2. AUTOCORRELACION")
print("Durbin-Watson:", round(dw, 4))
print("Breusch-Godfrey p-valor:", round(bg[1], 6))

# 3. Normalidad
jb = jarque_bera(residuos)

print("\n3. NORMALIDAD")
print("Jarque-Bera:", round(jb[0], 4))
print("p-valor:", round(jb[1], 6))

# 4. Errores estandar robustos HC3
modelo_robusto = modelo.get_robustcov_results(cov_type="HC3")

print("\n4. REGRESION CON ERRORES ESTANDAR ROBUSTOS HC3")
print(modelo_robusto.summary())

