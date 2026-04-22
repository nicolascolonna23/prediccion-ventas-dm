import pandas as pd
import numpy as np
from datetime import date
import base64, os, requests
from io import BytesIO

# 1. CARGAMOS TODA LA HISTORIA PARA QUE EL MODELO APRENDA BIEN
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
        return df
    except Exception as e:
        print(f"Error descargando {url}: {e}")
        return None

print("Descargando datos históricos...")
dfs = [descargar(u) for u in URLS]
dfs = [d for d in dfs if d is not None and len(d) > 0]
df_completo = pd.concat(dfs, ignore_index=True)

# Limpieza inicial
df_completo = df_completo.rename(columns={"Fecha de Emisión": "Fecha", "Monto Total": "Total"})
df_completo["Neto"] = pd.to_numeric(df_completo["Neto"], errors='coerce').fillna(0)
df_completo["Total"] = pd.to_numeric(df_completo["Total"], errors='coerce').fillna(0)

# Corrección de Netos 0 usando el Total (fundamental para 2025)
df_completo.loc[(df_completo["Neto"] == 0) & (df_completo["Total"] > 0), "Neto"] = df_completo["Total"] / 1.21

df_completo["Fecha"] = pd.to_datetime(df_completo["Fecha"], dayfirst=True, errors="coerce")
df_completo = df_completo.dropna(subset=["Fecha"])
df_completo["Anio"] = df_completo["Fecha"].dt.year
df_completo["Mes"]  = df_completo["Fecha"].dt.month

# Agrupación mensual NOMINAL de toda la historia
mensual_full = df_completo.groupby(["Anio", "Mes"])["Neto"].sum().reset_index()
mensual_full = mensual_full.sort_values(["Anio", "Mes"]).reset_index(drop=True)

# 2. ENTRENAMIENTO CON TODA LA DATA (Lags)
mensual_full["Lag1"] = mensual_full["Neto"].shift(1)
mensual_full["Lag2"] = mensual_full["Neto"].shift(2)
mensual_full["Lag3"] = mensual_full["Neto"].shift(3)
m_train = mensual_full.dropna().copy()

X = m_train[["Anio", "Mes", "Lag1", "Lag2", "Lag3"]].values
y = m_train["Neto"].values

X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X_std[X_std == 0] = 1
Xn = (X - X_mean) / X_std
y_mean, y_std = y.mean(), y.std()
yn = (y - y_mean) / y_std

A = np.c_[np.ones(len(Xn)), Xn]
coef, _, _, _ = np.linalg.lstsq(A, yn, rcond=None)

def predecir_nominal(anio, mes, l1, l2, l3):
    x = np.array([anio, mes, l1, l2, l3])
    xn = (x - X_mean) / X_std
    return (coef[0] + np.dot(coef[1:], xn)) * y_std + y_mean

# 3. PREDICCIONES FUTURAS
ultimo = mensual_full.iloc[-1]
lags = [float(ultimo["Neto"]), float(mensual_full.iloc[-2]["Neto"]), float(mensual_full.iloc[-3]["Neto"])]
predicciones = []
m_act, a_act = int(ultimo["Mes"]), int(ultimo["Anio"])

for i in range(1, 4):
    m_act += 1
    if m_act > 12: m_act = 1; a_act += 1
    p = predecir_nominal(a_act, m_act, lags[0], lags[1], lags[2])
    predicciones.append((a_act, m_act, p))
    lags = [p, lags[0], lags[1]]

# 4. RECORTE PARA EL DASHBOARD (Solo 2025 en adelante)
mensual_dash = mensual_full[mensual_full["Anio"] >= 2025].copy()

# --- GENERACIÓN DE DATOS PARA JS ---
nombres = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}

etiq_hist = [f"{nombres[int(r.Mes)]} {int(r.Anio)}" for _, r in mensual_dash.iterrows()]
netos_hist = [round(r.Neto / 1e6, 1) for _, r in mensual_dash.iterrows()]
preds_hist = [round(predecir_nominal(r.Anio, r.Mes, r.Lag1, r.Lag2, r.Lag3)/1e6, 1) for _, r in mensual_dash.iterrows()]

etiq_fut = [f"{nombres[m]} {a}" for a, m, p in predicciones]
preds_fut = [round(p / 1e6, 1) for a, m, p in predicciones]

etiquetas = etiq_hist + etiq_fut
netos_js  = "[" + ",".join(str(v) for v in netos_hist) + "," + ",".join("null" for _ in predicciones) + "]"
preds_js  = "[" + ",".join(str(v) for v in preds_hist) + "," + ",".join(str(v) for v in preds_fut) + "]"

# 5. HTML (Idem anterior pero con datos recortados)
hoy = date.today().strftime("%d/%m/%Y")
cards_html = "".join([f'<div class="card"><div class="card-label">{nombres[m]} {a}</div><div class="card-value">$ {p/1e6:,.1f} M</div><div class="card-sub">Nominal</div></div>' for a,m,p in predicciones])

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'><title>Ventas DM</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        body{{font-family:sans-serif;background:#f0f2f5;padding:20px}}
        .container{{max-width:1000px;margin:auto}}
        .header{{background:white;padding:20px;border-radius:10px;margin-bottom:20px;border-left:5px solid #1a4fa0}}
        .cards{{display:flex;gap:15px;margin-bottom:20px}}
        .card{{flex:1;background:white;padding:15px;border-radius:10px;text-align:center;box-shadow:0 2px 5px rgba(0,0,0,0.1)}}
        .box{{background:white;padding:20px;border-radius:10px;box-shadow:0 2px 5px rgba(0,0,0,0.1)}}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 style="color:#1a4fa0">DM Vencemos Distancias</h1>
            <p>Dashboard Nominal (Pesos Actuales) | Entrenado con historia 2021-2026</p>
        </div>
        <div class="cards">{cards_html}</div>
        <div class="box">
            <canvas id="chart" style="max-height:400px;"></canvas>
        </div>
    </div>
    <script>
        new Chart(document.getElementById('chart'), {{
            type: 'bar',
            data: {{
                labels: {etiquetas},
                datasets: [
                    {{ label: 'Neto Real Facturado (M)', data: {netos_js}, backgroundColor: '#1a4fa0', order: 2 }},
                    {{ label: 'Tendencia Modelo (M)', data: {preds_js}, type: 'line', borderColor: '#f28e2b', borderWidth: 2, order: 1 }}
                ]
            }},
            options: {{ scales: {{ y: {{ ticks: {{ callback: v => '$' + v + 'M' }} }} }} }}
        }});
    </script>
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)
print("index.html generado (Vista 2025+, Entrenamiento 2021+)")
