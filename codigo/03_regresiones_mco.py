
# PROYECTO FINAL DE ECONOMETRIA
# NVIDIA, NASDAQ-100 Y SOX
# Equipo: Annuar, Santiago y Mauricio

import pandas as pd
import statsmodels.api as sm
from pathlib import Path

# 1. Localizar la carpeta principal del proyecto
carpeta_proyecto = Path(__file__).resolve().parent.parent

# 2. Localizar la base de datos de Santiago
ruta_datos = carpeta_proyecto / "datos" / "processed" / "rendimientos_semanales.csv"

# 3. Cargar los datos
datos = pd.read_csv(ruta_datos, parse_dates=["fecha"])

# 4. Comprobar que se cargaron correctamente
print("PROYECTO NVIDIA - ECONOMETRIA")
print("Numero de observaciones:", len(datos))
print("\nPrimeras cinco observaciones:")
print(datos.head())


# --------------------------------------------------
# MODELO 1: NVIDIA EXPLICADA POR EL NASDAQ-100
# --------------------------------------------------

# Variable dependiente (Y)
Y = datos["r_NVDA"]

# Variable independiente (X)
X1 = datos[["r_NASDAQ"]]

# Agregar la constante beta_0
X1 = sm.add_constant(X1)

# Estimar el modelo por Minimos Cuadrados Ordinarios
modelo_1 = sm.OLS(Y, X1).fit()

# Mostrar los resultados completos
print("\n" + "=" * 55)
print("MODELO 1: NVIDIA VS NASDAQ-100")
print("=" * 55)
print(modelo_1.summary())


# =====================================================
# MODELO 2: NVIDIA EXPLICADA POR EL INDICE SOX
# =====================================================

# Variable dependiente: rendimiento semanal de NVIDIA
Y = datos["r_NVDA"]

# Variable independiente: rendimiento semanal del SOX
X2 = datos["r_SOX"]

# Agregar la constante beta_0
X2 = sm.add_constant(X2)

# Estimar el modelo por Minimos Cuadrados Ordinarios
modelo_2 = sm.OLS(Y, X2).fit()

# Mostrar los resultados
print("\n" + "=" * 55)
print("MODELO 2: NVIDIA VS SOX")
print("=" * 55)
print(modelo_2.summary())


# =====================================================
# MODELO 3: REGRESION MULTIPLE NVIDIA, NASDAQ Y SOX
# =====================================================

# Variable dependiente
Y = datos["r_NVDA"]

# Dos variables explicativas
X3 = datos[["r_NASDAQ", "r_SOX"]]

# Agregar constante
X3 = sm.add_constant(X3)

# Estimar regresion multiple por MCO
modelo_3 = sm.OLS(Y, X3).fit()

# Mostrar resultados
print("\n" + "=" * 55)
print("MODELO 3: NVIDIA VS NASDAQ-100 Y SOX")
print("=" * 55)
print(modelo_3.summary())


# =====================================================
# PRUEBAS F PARCIALES ENTRE MODELOS ANIDADOS
# =====================================================

# Prueba 1: ¿SOX aporta información adicional al Nasdaq?
# H0: beta_SOX = 0
f_sox, p_sox, gl_sox = modelo_3.compare_f_test(modelo_1)

print("\n" + "=" * 55)
print("PRUEBA F PARCIAL 1: AGREGAR SOX AL NASDAQ")
print("=" * 55)
print("F parcial:", round(f_sox, 4))
print("p-valor:", p_sox)
print("Numero de restricciones:", int(gl_sox))

# Prueba 2: ¿Nasdaq aporta información adicional al SOX?
# H0: beta_NASDAQ = 0
f_nasdaq, p_nasdaq, gl_nasdaq = modelo_3.compare_f_test(modelo_2)

print("\n" + "=" * 55)
print("PRUEBA F PARCIAL 2: AGREGAR NASDAQ AL SOX")
print("=" * 55)
print("F parcial:", round(f_nasdaq, 4))
print("p-valor:", p_nasdaq)
print("Numero de restricciones:", int(gl_nasdaq))

# Decisiones al 5% de significancia
alpha = 0.05

print("\nCONCLUSIONES:")
print(
    "Agregar SOX:",
    "Significativo" if p_sox < alpha else "No significativo"
)
print(
    "Agregar NASDAQ:",
    "Significativo" if p_nasdaq < alpha else "No significativo"
)

# =====================================================
# TABLA COMPARATIVA FINAL DE LOS TRES MODELOS
# =====================================================

from pathlib import Path

# Carpeta de resultados
carpeta_resultados = (
    Path(__file__).resolve().parent.parent / "resultados"
)
carpeta_resultados.mkdir(exist_ok=True)

modelos = {
    "Modelo 1 - Nasdaq": modelo_1,
    "Modelo 2 - SOX": modelo_2,
    "Modelo 3 - Ambos": modelo_3
}

filas = []

for nombre, modelo in modelos.items():
    filas.append({
        "Modelo": nombre,
        "Observaciones": int(modelo.nobs),
        "Beta_0": modelo.params.get("const"),
        "Beta_Nasdaq": modelo.params.get("r_NASDAQ"),
        "Beta_SOX": modelo.params.get("r_SOX"),
        "R2": modelo.rsquared,
        "R2_ajustado": modelo.rsquared_adj,
        "F_global": modelo.fvalue,
        "p_F_global": modelo.f_pvalue,
        "AIC": modelo.aic,
        "BIC": modelo.bic
    })

tabla = pd.DataFrame(filas)

# Guardar resultados sin perder precision
ruta_salida = carpeta_resultados / "comparacion_modelos.csv"
tabla.to_csv(ruta_salida, index=False)

print("\n" + "=" * 65)
print("TABLA COMPARATIVA FINAL")
print("=" * 65)
print(tabla.round(4).to_string(index=False))

print("\nArchivo guardado en:")
print(ruta_salida)


# =====================================================
# TABLA DE INFERENCIA: MCO Y ROBUSTEZ HC3
# =====================================================

filas_inferencia = []

for nombre, modelo in modelos.items():
    intervalos = modelo.conf_int()

    for variable in modelo.params.index:
        filas_inferencia.append({
            "Modelo": nombre,
            "Metodo": "MCO convencional",
            "Variable": variable,
            "Coeficiente": modelo.params[variable],
            "Error_estandar": modelo.bse[variable],
            "Estadistico_t": modelo.tvalues[variable],
            "p_valor": modelo.pvalues[variable],
            "IC_95_inferior": intervalos.loc[variable, 0],
            "IC_95_superior": intervalos.loc[variable, 1]
        })

# Inferencia robusta HC3 para el modelo multiple
modelo_3_hc3 = modelo_3.get_robustcov_results(
    cov_type="HC3"
)

for i, variable in enumerate(modelo_3.model.exog_names):
    filas_inferencia.append({
        "Modelo": "Modelo 3 - Ambos",
        "Metodo": "MCO robusto HC3",
        "Variable": variable,
        "Coeficiente": modelo_3_hc3.params[i],
        "Error_estandar": modelo_3_hc3.bse[i],
        "Estadistico_t": modelo_3_hc3.tvalues[i],
        "p_valor": modelo_3_hc3.pvalues[i],
        "IC_95_inferior": modelo_3_hc3.conf_int()[i, 0],
        "IC_95_superior": modelo_3_hc3.conf_int()[i, 1]
    })

tabla_inferencia = pd.DataFrame(filas_inferencia)

ruta_inferencia = (
    carpeta_resultados / "inferencia_modelos.csv"
)

tabla_inferencia.to_csv(
    ruta_inferencia,
    index=False
)

print("\n" + "=" * 65)
print("TABLA DE INFERENCIA ESTADISTICA")
print("=" * 65)

print(
    tabla_inferencia.round(4).to_string(index=False)
)

print("\nArchivo guardado en:")
print(ruta_inferencia)
