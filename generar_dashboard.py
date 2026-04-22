\\import pandas as pd
import numpy as np
from datetime import date
import base64, os, requests
from io import BytesIO

URLS = [
    "http://bi.sistemaexpreso.com.ar/reporte_cuenta_corriente_2021.xlsx",
    "http://bi.sistemaexpreso.com.ar/reporte_cuenta_corriente_2022.xlsx",
    "http://bi.sistemaexpreso.com.ar/reporte_cuenta_corriente_2023.xlsx",
    "http://bi.sistemaexpreso.com.ar/reporte_cuenta_corriente_2024.xlsx",
    "http://bi.sistemaexpreso.com.ar/reporte_cuenta_corriente_2025.xlsx",
    "http://bi.sistemaexpreso.com.ar/reporte_cuenta_corriente_2026.xlsx",
]

def descargar(url):
    try:
        r = requests.get(url, timeout=60)
        df = pd.read_excel(BytesIO(r.content))
        df.columns = [c.strip() for c in df.columns]
        if 'Total' not in df.columns and 'Monto Total' in df.columns:
            df = df.rename(columns={'Monto Total': 'Total'})
        if 'Neto' not in df.columns and 'Valor Neto' in df.columns:
            df = df.rename(columns={'Valor Neto': 'Neto'})
        return df
    except Exception as e:
        print(f"Error descargando {url}: {e}")
        return None

print("Descargando datos...")
dfs = [descargar(u) for u in URLS]
dfs = [d for d in dfs if d is not None and len(d) > 0]
df_completo = pd.concat(dfs, ignore_index=True)

df_completo = df_completo.rename(columns={"Fecha de Emisión": "Fecha"})
for col in ["Neto", "Total"]:
    if col not in df_completo.columns: df_completo[col] = 0
    df_completo[col] = pd.to_numeric(df_completo[col], errors='coerce').fillna(0)

# Corrección de Netos 0
df_completo.loc[(df_completo["Neto"] == 0) & (df_completo["Total"] > 0), "Neto"] = df_completo["Total"] / 1.21

df_completo["Fecha"] = pd.to_datetime(df_completo["Fecha"], dayfirst=True, errors="coerce")
df_completo = df_completo.dropna(subset=["Fecha"])
df_completo["Anio"] = df_completo["Fecha"].dt.year
df_completo["Mes"]  = df_completo["Fecha"].dt.month

# Agrupación mensual
mensual_full = df_completo.groupby(["Anio", "Mes"])["Neto"].sum().reset_index()
mensual_full = mensual_full.sort_values(["Anio", "Mes"]).reset_index(drop=True)

# --- EXCLUSIÓN DEL MES ACTUAL (Abierto) ---
hoy = date.today()
m_entrenamiento = mensual_full[~((mensual_full["Anio"] == hoy.year) & (mensual_full["Mes"] == hoy.month))].copy()

# --- MODELO CON ESTACIONALIDAD (VARIABLES DUMMY) ---
# Creamos columnas 0/1 para los meses (del 1 al 11, el 12 queda como base)
for m in range(1, 12):
    m_entrenamiento[f"Mes_{m}"] = (m_entrenamiento["Mes"] == m).astype(int)

m_entrenamiento["Lag1"] = m_entrenamiento["Neto"].shift(1)
m_entrenamiento["Lag2"] = m_entrenamiento["Neto"].shift(2)
m_entrenamiento["Lag3"] = m_entrenamiento["Neto"].shift(3)
m_train = m_entrenamiento.dropna().copy()

# Definimos las columnas que el modelo va a usar
columnas_estacionales = [f"Mes_{m}" for m in range(1, 12)]
columnas_x = ["Anio", "Lag1", "Lag2", "Lag3"] + columnas_estacionales

X = m_train[columnas_x].values
y = m_train["Neto"].values

# Normalización
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X_std[X_std == 0] = 1
Xn = (X - X_mean) / X_std
y_mean, y_std = y.mean(), y.std()
yn = (y - y_mean) / y_std

A = np.c_[np.ones(len(Xn)), Xn]
coef, _, _, _ = np.linalg.lstsq(A, yn, rcond=None)

def predecir_estacional(anio, mes, l1, l2, l3):
    # Crear vector de dummies para el mes solicitado
    dummies = [1 if mes == m else 0 for m in range(1, 12)]
    x = np.array([anio, l1, l2, l3] + dummies)
    xn = (x - X_mean) / X_std
    return (coef[0] + np.dot(coef[1:], xn)) * y_std + y_mean

# --- PREDICCIONES FUTURAS ---
ultimo_cerrado = m_entrenamiento.iloc[-1]
lags = [float(ultimo_cerrado["Neto"]), float(m_entrenamiento.iloc[-2]["Neto"]), float(m_entrenamiento.iloc[-3]["Neto"])]
predicciones = []
m_act, a_act = int(ultimo_cerrado["Mes"]), int(ultimo_cerrado["Anio"])

for i in range(1, 4):
    m_act += 1
    if m_act > 12: m_act = 1; a_act += 1
    p = predecir_estacional(a_act, m_act, lags[0], lags[1], lags[2])
    predicciones.append((a_act, m_act, p))
    lags = [p, lags[0], lags[1]]

# --- DASHBOARD VISTA 2025+ ---
mensual_dash = mensual_full[mensual_full["Anio"] >= 2025].copy()
nombres = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}

etiq_hist = [f"{nombres[int(r.Mes)]} {int(r.Anio)}" for _, r in mensual_dash.iterrows()]
netos_hist = [round(r.Neto / 1e6, 1) for _, r in mensual_dash.iterrows()]

# Limpieza de etiqueta duplicada de Abril (mes abierto)
if len(etiq_hist) > 0:
    ultima_f = mensual_dash.iloc[-1]
    if (hoy.month == ultima_f["Mes"]) and (hoy.year == ultima_f["Anio"]):
        etiq_hist = etiq_hist[:-1]

# Tendencia histórica con estacionalidad
preds_hist = []
mensual_dash["L1"] = mensual_dash["Neto"].shift(1)
mensual_dash["L2"] = mensual_dash["Neto"].shift(2)
mensual_dash["L3"] = mensual_dash["Neto"].shift(3)

for _, r in mensual_dash.iterrows():
    if pd.notnull(r.L3):
        preds_hist.append(round(predecir_estacional(r.Anio, r.Mes, r.L1, r.L2, r.L3)/1e6, 1))
    else:
        preds_hist.append(None)

etiq_fut = [f"{nombres[m]} {a}" for a, m, p in predicciones]
preds_fut = [round(p / 1e6, 1) for a, m, p in predicciones]

etiquetas = etiq_hist + etiq_fut
netos_js  = "[" + ",".join(str(v) for v in netos_hist) + "," + ",".join("null" for _ in predicciones) + "]"
preds_js  = "[" + ",".join(str(v) if v is not None else "null" for v in preds_hist) + "," + ",".join(str(v) for v in preds_fut) + "]"

# HTML
cards_html = "".join([f'<div class="card"><div class="card-label">{nombres[m]} {a}</div><div class="card-value">$ {p/1e6:,.1f} M</div></div>' for a,m,p in predicciones])

html_content = f"""
<!DOCTYPE html><html><head><meta charset='utf-8'><title>DM Dashboard Estacional</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
    body{{font-family:sans-serif;background:#f8f9fa;padding:20px}}
    .container{{max-width:950px;margin:auto}}
    .header{{background:#fff;padding:15px;border-radius:10px;margin-bottom:20px;border-left:5px solid #1a4fa0}}
    .cards{{display:flex;gap:15px;margin-bottom:20px}}
    .card{{flex:1;background:#fff;padding:15px;border-radius:10px;text-align:center;box-shadow:0 2px 4px rgba(0,0,0,0.05)}}
    .card-label{{font-size:11px;color:#888;text-transform:uppercase}}
    .card-value{{font-size:22px;font-weight:bold;color:#1a4fa0}}
</style></head><body>
<div class="container">
    <div class="header"><h1>DM Vencemos Distancias</h1><p>Modelo con <b>Inteligencia Estacional</b> (Analiza comportamiento mes a mes desde 2021)</p></div>
    <div class="cards">{cards_html}</div>
    <div style="background:#fff;padding:20px;border-radius:10px;box-shadow:0 2px 5px rgba(0,0,0,0.05);"><canvas id="chart"></canvas></div>
</div>
<script>
    new Chart(document.getElementById('chart'), {{
        type: 'bar',
        data: {{
            labels: {etiquetas},
            datasets: [
                {{ label: 'Ventas Reales (M)', data: {netos_js}, backgroundColor: '#1a4fa0', borderRadius: 4 }},
                {{ label: 'Predicción Estacional (M)', data: {preds_js}, type: 'line', borderColor: '#f28e2b', tension: 0.3, pointRadius: 4 }}
            ]
        }},
        options: {{
            plugins: {{ tooltip: {{ callbacks: {{ label: ctx => '$ ' + ctx.parsed.y + ' M' }} }} }},
            scales: {{ y: {{ ticks: {{ callback: v => '$' + v + 'M' }} }} }}
        }}
    }});
</script></body></html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)
print("Dashboard con estacionalidad generado.")
