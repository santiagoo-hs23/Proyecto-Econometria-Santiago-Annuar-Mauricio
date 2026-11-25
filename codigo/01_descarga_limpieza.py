import os
import numpy as np
import pandas as pd
import yfinance as yf

# 1. Parámetros del estudio
tickers = ['NVDA', '^NDX', '^SOX']
start_date = '2021-11-01'
end_date = '2026-10-31'

print("=== PASO 1: DESCARGA DE DATOS HISTÓRICOS ===")
# Precios ajustados por splits y dividendos
df_raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)

# Extraer precios de cierre
if isinstance(df_raw.columns, pd.MultiIndex):
    precios_diarios = df_raw['Close'] if 'Close' in df_raw.columns.levels[0] else df_raw
else:
    precios_diarios = df_raw

# Reordenar y nombrar columnas de trabajo
precios_diarios = precios_diarios[['NVDA', '^NDX', '^SOX']].rename(
    columns={'^NDX': 'NASDAQ', '^SOX': 'SOX'}
)

# Guardar respaldo original en datos/raw/
os.makedirs('datos/raw', exist_ok=True)
precios_diarios.to_csv('datos/raw/precios_originales.csv')
print("✓ Precios diarios originales guardados en 'datos/raw/precios_originales.csv'\n")

print("=== PASO 2: LIMPIEZA Y TRANSFORMACIÓN SEMANAL ===")

# Revisión de nulos en datos diarios
nulos_diarios = precios_diarios.isnull().sum().sum()
print(f"Valores nulos en serie diaria: {nulos_diarios}")

# Alineación temporal diaria (se eliminan días festivos dispares)
precios_diarios = precios_diarios.dropna()

# Conversión a precios cierres semanales (último cierre de cada semana)
precios_semanales = precios_diarios.resample('W-FRI').last().dropna()

# Cálculo de rendimientos logarítmicos semanales (%)
rendimientos = np.log(precios_semanales / precios_semanales.shift(1)) * 100

# Eliminar primera fila con NaN tras el retardo
rendimientos = rendimientos.dropna().reset_index()

# Formato final de columnas requerido para el equipo
rendimientos.columns = ['fecha', 'r_NVDA', 'r_NASDAQ', 'r_SOX']
rendimientos['fecha'] = rendimientos['fecha'].dt.strftime('%Y-%m-%d')

# Verificación de duplicados
duplicados = rendimientos.duplicated(subset=['fecha']).sum()
print(f"Filas duplicadas: {duplicados}")

print("\n=== INFORMACIÓN DE LA BASE FINAL DE RENDIMIENTOS ===")
print(f"Número total de observaciones semanales (N): {len(rendimientos)}")
print(f"Fecha inicial: {rendimientos['fecha'].iloc[0]}")
print(f"Fecha final:   {rendimientos['fecha'].iloc[-1]}")
print("\nVista previa de las primeras 5 semanas:")
print(rendimientos.head())

# Guardar la base procesada en datos/processed/
os.makedirs('datos/processed', exist_ok=True)
path_processed = 'datos/processed/rendimientos_semanales.csv'
rendimientos.to_csv(path_processed, index=False)

print(f"\n✓ Base limpia guardada con éxito en '{path_processed}'")