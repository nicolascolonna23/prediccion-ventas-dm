import pandas as pd
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

df_completo.loc[(df_completo["Neto"] == 0) & (df_completo["Total"] > 0), "Neto"] = df_completo["Total"] / 1.21

df_completo["Fecha"] = pd.to_datetime(df_completo["Fecha"], dayfirst=True, errors="coerce")
df_completo = df_completo.dropna(subset=["Fecha"])
df_completo["Anio"] = df_completo["Fecha"].dt.year
df_completo["Mes"]  = df_completo["Fecha"].dt.month

mensual_full = df_completo.groupby(["Anio", "Mes"])["Neto"].sum().reset_index()
mensual_full = mensual_full.sort_values(["Anio", "Mes"]).reset_index(drop=True)

hoy = date.today()
m_entrenamiento = mensual_full[~((mensual_full["Anio"] == hoy.year) & (mensual_full["Mes"] == hoy.month))].copy()

for m in range(1, 12):
    m_entrenamiento[f"Mes_{m}"] = (m_entrenamiento["Mes"] == m).astype(int)

m_entrenamiento["Lag1"] = m_entrenamiento["Neto"].shift(1)
m_entrenamiento["Lag2"] = m_entrenamiento["Neto"].shift(2)
m_entrenamiento["Lag3"] = m_entrenamiento["Neto"].shift(3)
m_train = m_entrenamiento.dropna().copy()

columnas_estacionales = [f"Mes_{m}" for m in range(1, 12)]
columnas_x = ["Anio", "Lag1", "Lag2", "Lag3"] + columnas_estacionales

X = m_train[columnas_x].values
y = m_train["Neto"].values
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X_std[X_std == 0] = 1
Xn = (X - X_mean) / X_std
y_mean, y_std = y.mean(), y.std()
yn = (y - y_mean) / y_std
A = np.c_[np.ones(len(Xn)), Xn]
coef, _, _, _ = np.linalg.lstsq(A, yn, rcond=None)

def predecir_estacional(anio, mes, l1, l2, l3):
    dummies = [1 if mes == m else 0 for m in range(1, 12)]
    x = np.array([anio, l1, l2, l3] + dummies)
    xn = (x - X_mean) / X_std
    return (coef[0] + np.dot(coef[1:], xn)) * y_std + y_mean

# --- PREDICCIONES ---
ultimo_cerrado = m_entrenamiento.iloc[-1]
lags = [float(ultimo_cerrado["Neto"]), float(m_entrenamiento.iloc[-2]["Neto"]), float(m_entrenamiento.iloc[-3]["Neto"])]
predicciones_raw = []
m_act, a_act = int(ultimo_cerrado["Mes"]), int(ultimo_cerrado["Anio"])
pasos = 12 - m_act if a_act == hoy.year else 12

for i in range(pasos):
    m_act += 1
    if m_act > 12: m_act = 1; a_act += 1
    p = predecir_estacional(a_act, m_act, lags[0], lags[1], lags[2])
    predicciones_raw.append({"Anio": int(a_act), "Mes": int(m_act), "Neto": p})
    lags = [p, lags[0], lags[1]]

# --- DATA JS ---
nombres_meses = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}
data_js = []
for idx, r in mensual_full.iterrows():
    l1 = mensual_full.loc[idx-1, "Neto"] if idx > 0 else None
    l2 = mensual_full.loc[idx-2, "Neto"] if idx > 1 else None
    l3 = mensual_full.loc[idx-3, "Neto"] if idx > 2 else None
    p_val = round(predecir_estacional(r['Anio'], r['Mes'], l1, l2, l3)/1e6, 2) if l3 is not None else None
    data_js.append({"label": f"{nombres_meses[int(r['Mes'])]} {int(r['Anio'])}", "anio": int(r['Anio']), "mes": int(r['Mes']), "neto": round(r['Neto']/1e6, 2), "pred": p_val})

for p in predicciones_raw:
    data_js.append({"label": f"{nombres_meses[p['Mes']]} {p['Anio']}", "anio": p['Anio'], "mes": p['Mes'], "neto": None, "pred": round(p['Neto']/1e6, 2)})

years_options = sorted(list(mensual_full['Anio'].unique()))
cards_html = "".join([f'<div class="card"><div class="card-label">{nombres_meses[p["Mes"]]} {p["Anio"]}</div><div class="card-value">$ {p["Neto"]/1e6:,.1f} M</div></div>' for p in predicciones_raw[:3]])

html_content = f"""
<!DOCTYPE html><html><head><meta charset='utf-8'><title>DM Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
    body{{font-family:sans-serif;background:#f4f7f6;padding:20px;color:#333}}
    .container{{max-width:1100px;margin:auto}}
    .header{{background:#fff;padding:20px;border-radius:12px;margin-bottom:20px;display:flex;justify-content:space-between;align-items:center;box-shadow:0 2px 5px rgba(0,0,0,0.05)}}
    .logo{{max-height:60px}}
    .cards{{display:flex;gap:15px;margin-bottom:20px}}
    .card{{flex:1;background:#fff;padding:15px;border-radius:12px;text-align:center;border-bottom:4px solid #1a4fa0;box-shadow:0 2px 4px rgba(0,0,0,0.05)}}
    .card-label{{font-size:11px;color:#888;text-transform:uppercase}}
    .card-value{{font-size:22px;font-weight:bold;color:#1a4fa0}}
    .filters{{background:#fff;padding:20px;border-radius:12px;margin-bottom:20px;display:flex;gap:20px;box-shadow:0 2px 5px rgba(0,0,0,0.05)}}
    .filter-group{{flex:1}}
    select{{width:100%;padding:8px;border-radius:6px;border:1px solid #ccc;height:80px}}
    .chart-box{{background:#fff;padding:20px;border-radius:12px;box-shadow:0 2px 5px rgba(0,0,0,0.05)}}
    .info{{margin-top:20px;background:#eef2f7;padding:20px;border-radius:12px;font-size:14px;line-height:1.6}}
</style></head><body>
<div class="container">
    <div class="header">
        <img src="logo_dm.png" alt="DM" class="logo">
        <div style="text-align:right"><h1>Dashboard Interactivo</h1><p>Ventas Nominales Proyectadas</p></div>
    </div>
    <div class="cards">{cards_html}</div>
    <div class="filters">
        <div class="filter-group"><label><b>Años (Multiselección):</b></label><br>
            <select id="yF" multiple onchange="uC()">
                {"".join([f'<option value="{a}" selected>{a}</option>' for a in years_options])}
            </select>
        </div>
        <div class="filter-group"><label><b>Meses:</b></label><br>
            <select id="mF" multiple onchange="uC()">
                {"".join([f'<option value="{i}" selected>{n}</option>' for i, n in nombres_meses.items()])}
            </select>
        </div>
    </div>
    <div class="chart-box"><canvas id="chart"></canvas></div>
    <div class="info">
        <h3>¿Cómo funciona este modelo?</h3>
        <p>Este sistema utiliza <b>IA Estacional</b>: analiza los últimos 5 años de historia para entender que cada mes tiene un comportamiento único. Se basa en "Lags" (inercia de ventas de los últimos 3 meses cerrados) y variables estacionales para proyectar el resto del año. El mes en curso se excluye del entrenamiento para no distorsionar la tendencia.</p>
    </div>
</div>
<script>
    const d = {data_js}; let c;
    function uC() {{
        const sY = Array.from(document.getElementById('yF').selectedOptions).map(o=>parseInt(o.value));
        const sM = Array.from(document.getElementById('mF').selectedOptions).map(o=>parseInt(o.value));
        const f = d.filter(x => sY.includes(x.anio) && sM.includes(x.mes));
        if(c) c.destroy();
        c = new Chart(document.getElementById('chart'), {{
            type: 'bar',
            data: {{
                labels: f.map(x=>x.label),
                datasets: [
                    {{ label: 'Ventas Reales (M)', data: f.map(x=>x.neto), backgroundColor: '#1a4fa0', borderRadius: 5 }},
                    {{ label: 'Proyección IA (M)', data: f.map(x=>x.pred), type: 'line', borderColor: '#f28e2b', tension: 0.3 }}
                ]
            }},
            options: {{ scales: {{ y: {{ beginAtZero: true, ticks: {{ callback: v => '$'+v+'M' }} }} }} }}
        }});
    }}
    uC();
</script></body></html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)
