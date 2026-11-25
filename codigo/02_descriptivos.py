import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración estética global para las gráficas
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# Crear carpeta de salidas si no existe
os.makedirs('graficas', exist_ok=True)
os.makedirs('datos/processed', exist_ok=True)

# 1. CARGAR DATOS
df_ret = pd.read_csv('datos/processed/rendimientos_semanales.csv')

# Cargar precios originales ajustados
df_prices = pd.read_csv('datos/raw/precios_originales.csv', index_col=0)
df_prices.index = pd.to_datetime(df_prices.index)

# Estandarizar columnas de precios
if 'NDX' in df_prices.columns:
    df_prices = df_prices.rename(columns={'NDX': 'NASDAQ'})
df_prices = df_prices[['NVDA', 'NASDAQ', 'SOX']].dropna()

# 2. ESTADÍSTICA DESCRIPTIVA (ENTREGABLE C & E5)
cols_ret = ['r_NVDA', 'r_NASDAQ', 'r_SOX']
stats = df_ret[cols_ret].describe(percentiles=[0.25, 0.50, 0.75]).T

# Renombrar columnas para formato académico
stats = stats[['count', 'mean', '50%', 'std', 'min', '25%', '75%', 'max']]
stats.columns = ['N', 'Media (%)', 'Mediana (%)', 'Desv. Est. (%)', 'Mínimo (%)', 'Q1 25% (%)', 'Q3 75% (%)', 'Máximo (%)']

# Guardar tabla de estadísticos descriptivos
stats.to_csv('datos/processed/tabla_estadisticos_descriptivos.csv')

print("=== TABLA DE ESTADÍSTICOS DESCRIPTIVOS (RENDIMIENTOS SEMANALES %) ===")
print(stats.round(4))
print("\n✓ Tabla guardada en 'datos/processed/tabla_estadisticos_descriptivos.csv'\n")

# MATRIZ DE CORRELACIÓN (ENTREGABLE E6)
corr_matrix = df_ret[cols_ret].corr()
corr_matrix.to_csv('datos/processed/matriz_correlacion.csv')

print("=== MATRIZ DE CORRELACIÓN DE PEARSON ===")
print(corr_matrix.round(4))
print("\n✓ Matriz guardada en 'datos/processed/matriz_correlacion.csv'\n")

# 3. GENERACIÓN DE GRÁFICAS (ENTREGABLE D)
periodo_str = f"Periodo: {df_ret['fecha'].iloc[0]} a {df_ret['fecha'].iloc[-1]}"

# --- GRÁFICA 1: EVOLUCIÓN HISTÓRICA DE PRECIOS NORMALIZADOS (BASE 100) ---
fig, ax = plt.subplots(figsize=(10, 5))
norm_prices = (df_prices / df_prices.iloc[0]) * 100
ax.plot(norm_prices.index, norm_prices['NVDA'], label='NVIDIA (NVDA)', color='#76B900', linewidth=2)
ax.plot(norm_prices.index, norm_prices['NASDAQ'], label='Nasdaq-100 (^NDX)', color='#004B87', linewidth=1.5)
ax.plot(norm_prices.index, norm_prices['SOX'], label='PHLX Semiconductor (^SOX)', color='#E06D53', linewidth=1.5)
ax.set_title(f'Evolución Histórica Comparativa de Precios (Base 100 = Nov 2021)\n{periodo_str}', fontsize=12, fontweight='bold')
ax.set_ylabel('Índice de Precio (Base 100)', fontsize=10)
ax.set_xlabel('Fecha', fontsize=10)
ax.legend(frameon=True, facecolor='white')
plt.tight_layout()
plt.savefig('graficas/01_precios_historicos.png', dpi=300)
plt.close()

# --- GRÁFICA 2: RENDIMIENTOS SEMANALES EN EL TIEMPO ---
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
fechas = pd.to_datetime(df_ret['fecha'])

axes[0].plot(fechas, df_ret['r_NVDA'], color='#76B900', linewidth=1)
axes[0].set_title('Rendimientos Semanales: NVIDIA (r_NVDA)', fontsize=10, fontweight='bold')
axes[0].set_ylabel('Rendimiento (%)')

axes[1].plot(fechas, df_ret['r_NASDAQ'], color='#004B87', linewidth=1)
axes[1].set_title('Rendimientos Semanales: Nasdaq-100 (r_NASDAQ)', fontsize=10, fontweight='bold')
axes[1].set_ylabel('Rendimiento (%)')

axes[2].plot(fechas, df_ret['r_SOX'], color='#E06D53', linewidth=1)
axes[2].set_title('Rendimientos Semanales: PHLX Semiconductor (r_SOX)', fontsize=10, fontweight='bold')
axes[2].set_ylabel('Rendimiento (%)')
axes[2].set_xlabel('Fecha', fontsize=10)

fig.suptitle(f'Series de Tiempo de Rendimientos Logarítmicos Semanales\n{periodo_str}', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('graficas/02_rendimientos_semanales.png', dpi=300)
plt.close()

# --- GRÁFICA 3: HISTOGRAMAS Y DISTRIBUCIONES ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

sns.histplot(df_ret['r_NVDA'], kde=True, color='#76B900', ax=axes[0], bins=25)
axes[0].set_title('Distribución r_NVDA', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Rendimiento (%)')
axes[0].set_ylabel('Frecuencia')

sns.histplot(df_ret['r_NASDAQ'], kde=True, color='#004B87', ax=axes[1], bins=25)
axes[1].set_title('Distribución r_NASDAQ', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Rendimiento (%)')
axes[1].set_ylabel('Frecuencia')

sns.histplot(df_ret['r_SOX'], kde=True, color='#E06D53', ax=axes[2], bins=25)
axes[2].set_title('Distribución r_SOX', fontsize=11, fontweight='bold')
axes[2].set_xlabel('Rendimiento (%)')
axes[2].set_ylabel('Frecuencia')

fig.suptitle(f'Histogramas de Rendimientos Semanales con Estimación de Densidad Kernel (KDE)\n{periodo_str}', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('graficas/03_histogramas_rendimientos.png', dpi=300)
plt.close()

# --- GRÁFICA 4: DIAGRAMAS DE DISPERSIÓN (NVDA vs NDX, NVDA vs SOX) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.regplot(data=df_ret, x='r_NASDAQ', y='r_NVDA', ax=axes[0],
            color='#004B87', scatter_kws={'alpha': 0.6}, line_kws={'color': 'red', 'linewidth': 2})
axes[0].set_title('r_NVDA vs r_NASDAQ', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Rendimiento Nasdaq-100 (%)')
axes[0].set_ylabel('Rendimiento NVIDIA (%)')

sns.regplot(data=df_ret, x='r_SOX', y='r_NVDA', ax=axes[1],
            color='#E06D53', scatter_kws={'alpha': 0.6}, line_kws={'color': 'red', 'linewidth': 2})
axes[1].set_title('r_NVDA vs r_SOX', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Rendimiento PHLX Semiconductor (%)')
axes[1].set_ylabel('Rendimiento NVIDIA (%)')

fig.suptitle(f'Diagramas de Dispersión y Ajuste de Regresión Simple\n{periodo_str}', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('graficas/04_scatter_nvda_indices.png', dpi=300)
plt.close()

# --- GRÁFICA 5: MATRIZ DE CORRELACIÓN (HEATMAP) ---
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, fmt='.4f', cmap='coolwarm', vmin=-1, vmax=1,
            cbar_kws={'label': 'Coeficiente de Correlación de Pearson'}, ax=ax, square=True)
ax.set_title(f'Matriz de Correlación de Pearson\n{periodo_str}', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('graficas/05_matriz_correlacion.png', dpi=300)
plt.close()

print("✓ Las 5 gráficas de alta resolución se exportaron exitosamente a la carpeta 'graficas/'")