import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib as mpl
import seaborn as sns
import statsmodels as sm

print("=== VERIFICACIÓN DE ENTORNO ECONOMÉTRICO ===")
print(f"yfinance version:   {yf.__version__}")
print(f"pandas version:     {pd.__version__}")
print(f"numpy version:      {np.__version__}")
print(f"matplotlib version: {mpl.__version__}")
print(f"seaborn version:    {sns.__version__}")
print(f"statsmodels version:{sm.__version__}")
print("==========================================")
print("¡El entorno está listo para trabajar!")
