import pandas as pd
import numpy as np
from datetime import date
import base64, os, requests
from io import BytesIO

# URLs de los reportes históricos
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
        # Parche de compatibilidad de nombres
        if 'Monto Total' in df.columns: df = df.rename(columns={'Monto Total': 'Total'})
        if 'Valor Neto' in df.columns: df = df.rename(columns={'Valor Neto': 'Neto'})
        return df
    except Exception as e:
        print(f"Error descargando {url}: {e}")
        return None

print("Descargando datos...")
dfs = [descargar(u) for u in URLS]
dfs = [d for d in dfs if d is not None and len(d) > 0]
df_completo = pd.concat(dfs, ignore_index=True)

# Limpieza y normalización
df_completo = df_completo.rename(columns={"Fecha de Emisión": "Fecha"})
for col in ["Neto", "Total"]:
    if col not in df_completo.columns: df_completo[col] = 0
    df_completo[col] = pd.to_numeric(df_completo[col], errors='coerce').fillna(0)

# Corrección de Netos 0 (Cálculo estimado basado en Monto Total)
df_completo.loc[(df_completo["Neto"] == 0) & (df_completo["Total"] > 0), "Neto"] = df_completo["Total"] / 1.21

df_completo["Fecha"] = pd.to_datetime(df_completo["Fecha"], dayfirst=True, errors="coerce")
df_completo = df_completo.dropna(subset=["Fecha"])
df_completo["Anio"] = df_completo["Fecha"].dt.year
df_completo["Mes"]  = df_completo["Fecha"].dt.month

# Agrupación mensual nominal
mensual_full = df_completo.groupby(["Anio", "Mes"])["Neto"].sum().reset_index()
mensual_full = mensual_full.sort_values(["Anio", "Mes"]).reset_index(drop=True)

# --- MODELO DE ENTRENAMIENTO (Excluyendo mes actual abierto) ---
hoy = date.today()
m_entrenamiento = mensual_full[~((mensual_full["Anio"] == hoy.year) & (mensual_full["Mes"] == hoy.month))].copy()

# Variables estacionales (Dummies)
for m in range(1, 12):
    m_entrenamiento[f"M_{m}"] = (m_entrenamiento["Mes"] == m).astype(int)

m_entrenamiento["L1"] = m_entrenamiento["Neto"].shift(1)
m_entrenamiento["L2"] = m_entrenamiento["Neto"].shift(2)
m_entrenamiento["L3"] = m_entrenamiento["Neto"].shift(3)
m_train = m_entrenamiento.dropna().copy()

# Entrenamiento
cols_x = ["Anio", "L1", "L2", "L3"] + [f"M_{m}" for m in range(1, 12)]
X = m_train[cols_x].values
y = m_train["Neto"].values
X_m, X_s = X.mean(axis=0), X.std(axis=0)
X_s[X_s == 0] = 1
Xn, y_m, y_s = (X - X_m) / X_s, y.mean(), y.std()
yn = (y - y_m) / y_s
A = np.c_[np.ones(len(Xn)), Xn]
coef, _, _, _ = np.linalg.lstsq(A, yn, rcond=None)

def predecir(a, m, l1, l2, l3):
    dums = [1 if m == i else 0 for i in range(1, 12)]
    xn = (np.array([a, l1, l2, l3] + dums) - X_m) / X_s
    return (coef[0] + np.dot(coef[1:], xn)) * y_s + y_m

# --- GENERACIÓN DE PREDICCIONES (Hasta fin de año) ---
u_c = m_entrenamiento.iloc[-1]
lags = [float(u_c["Neto"]), float(m_entrenamiento.iloc[-2]["Neto"]), float(m_entrenamiento.iloc[-3]["Neto"])]
predicciones = []
m_a, a_a = int(u_c["Mes"]), int(u_c["Anio"])
pasos = (12 - m_a) if a_a == hoy.year else 12

for i in range(pasos):
    m_a += 1
    if m_a > 12: m_a = 1; a_a += 1
    p_f = predecir(a_a, m_a, lags[0], lags[1], lags[2])
    predicciones.append({"a": a_a, "m": m_a, "p": p_f})
    lags = [p_f, lags[0], lags[1]]

# --- DASHBOARD (VISTA DESDE 2024) ---
mensual_dash = mensual_full[mensual_full["Anio"] >= 2024].copy()
nombres = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}

etiq_hist = [f"{nombres[int(r.Mes)]} {int(r.Anio)}" for _, r in mensual_dash.iterrows()]
netos_hist = [round(r.Neto / 1e6, 1) for _, r in mensual_dash.iterrows()]

# Evitar etiqueta duplicada del mes actual
if len(etiq_hist) > 0 and (hoy.month == mensual_dash.iloc[-1]["Mes"]) and (hoy.year == mensual_dash.iloc[-1]["Anio"]):
    etiq_hist = etiq_hist[:-1]

# Tendencia histórica
mensual_dash["L1"] = mensual_dash["Neto"].shift(1)
mensual_dash["L2"] = mensual_dash["Neto"].shift(2)
mensual_dash["L3"] = mensual_dash["Neto"].shift(3)
preds_hist = [round(predecir(r.Anio, r.Mes, r.L1, r.L2, r.L3)/1e6, 1) if pd.notnull(r.L3) else None for _, r in mensual_dash.iterrows()]

etiq_fut = [f"{nombres[p['m']]} {p['a']}" for p in predicciones]
preds_fut = [round(p['p'] / 1e6, 1) for p in predicciones]

etiquetas = etiq_hist + etiq_fut
netos_js  = "[" + ",".join(str(v) for v in netos_hist) + "," + ",".join("null" for _ in predicciones) + "]"
preds_js  = "[" + ",".join(str(v) if v is not None else "null" for v in preds_hist) + "," + ",".join(str(v) for v in preds_fut) + "]"

# HTML
cards_html = "".join([f'<div class="card"><div class="card-label">{nombres[p["m"]]} {p["a"]}</div><div class="card-value">$ {p["p"]/1e6:,.1f} M</div></div>' for p in predicciones[:3]])

html_content = f"""
<!DOCTYPE html><html><head><meta charset='utf-8'><title>DM Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
    body{{font-family:sans-serif;background:#f8f9fa;padding:20px;color:#333}}
    .container{{max-width:1000px;margin:auto}}
    .header{{background:#fff;padding:20px;border-radius:12px;margin-bottom:20px;text-align:center;box-shadow:0 2px 5px rgba(0,0,0,0.05)}}
    .logo{{max-height:60px;margin-bottom:10px}}
    .cards{{display:flex;gap:15px;margin-bottom:20px}}
    .card{{flex:1;background:#fff;padding:15px;border-radius:12px;text-align:center;border-bottom:4px solid #1a4fa0;box-shadow:0 2px 4px rgba(0,0,0,0.05)}}
    .card-label{{font-size:11px;color:#888;text-transform:uppercase}}
    .card-value{{font-size:22px;font-weight:bold;color:#1a4fa0;margin-top:5px}}
    .box{{background:#fff;padding:25px;border-radius:12px;box-shadow:0 2px 5px rgba(0,0,0,0.05)}}
    .info{{margin-top:20px;background:#eef2f7;padding:20px;border-radius:12px;font-size:14px;line-height:1.6}}
</style></head><body>
<div class="container">
    <div class="header"><img src="logo_dm.png" class="logo"><h1>DM Vencemos Distancias</h1><p>Proyecciones Nominales | Vista 2024 - 2026</p></div>
    <div class="cards">{cards_html}</div>
    <div class="box"><canvas id="chart"></canvas></div>
    <div class="info">
        <h3>¿Cómo funciona el modelo?</h3>
        <p>El sistema utiliza <b>IA Estacional</b>: analiza los últimos 5 años de historia para entender el comportamiento de cada mes. Se basa en la inercia de ventas de los últimos 3 meses cerrados (Lags) para proyectar el futuro, excluyendo el mes en curso para no distorsionar la tendencia real.</p>
    </div>
</div>
<script>
    new Chart(document.getElementById('chart'), {{
        type: 'bar',
        data: {{
            labels: {etiquetas},
            datasets: [
                {{ label: 'Ventas Reales (M)', data: {netos_js}, backgroundColor: '#1a4fa0', borderRadius: 5 }},
                {{ label: 'Tendencia IA (M)', data: {preds_js}, type: 'line', borderColor: '#f28e2b', tension: 0.3, pointRadius: 4 }}
            ]
        }},
        options: {{
            scales: {{ y: {{ ticks: {{ callback: v => '$' + v + 'M' }} }} }},
            plugins: {{ tooltip: {{ callbacks: {{ label: ctx => '$ ' + ctx.parsed.y + ' M' }} }} }}
        }}
    }});
</script></body></html>
"""
with open("index.html", "w", encoding="utf-8") as f: f.write(html_content)
