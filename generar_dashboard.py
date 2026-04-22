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

# Variables Dummy para Estacionalidad
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

# --- PREDICCIONES HASTA FIN DE AÑO ---
ultimo_cerrado = m_entrenamiento.iloc[-1]
lags = [float(ultimo_cerrado["Neto"]), float(m_entrenamiento.iloc[-2]["Neto"]), float(m_entrenamiento.iloc[-3]["Neto"])]
predicciones_raw = []
m_act, a_act = int(ultimo_cerrado["Mes"]), int(ultimo_cerrado["Anio"])
pasos = (12 - m_act) if a_act == hoy.year else 12 

for i in range(pasos):
    m_act += 1
    if m_act > 12: m_act = 1; a_act += 1
    p = predecir_estacional(a_act, m_act, lags[0], lags[1], lags[2])
    predicciones_raw.append({"Anio": int(a_act), "Mes": int(m_act), "Neto": round(p, 2), "Tipo": "Predicción"})
    lags = [p, lags[0], lags[1]]

# --- PREPARACIÓN DE DATA COMPLETA PARA JS ---
nombres_meses = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}

historico_js = []
for _, r in mensual_full.iterrows():
    # Calcular tendencia para cada punto histórico
    lag1 = mensual_full.loc[_-1, "Neto"] if _ > 0 else None
    lag2 = mensual_full.loc[_-2, "Neto"] if _ > 1 else None
    lag3 = mensual_full.loc[_-3, "Neto"] if _ > 2 else None
    
    pred_val = None
    if lag3 is not None:
        pred_val = round(predecir_estacional(r["Anio"], r["Mes"], lag1, lag2, lag3) / 1e6, 2)

    historico_js.append({
        "label": f"{nombres_meses[int(r['Mes'])]} {int(r['Anio'])}",
        "anio": int(r['Anio']),
        "mes": int(r['Mes']),
        "neto": round(r['Neto'] / 1e6, 2),
        "pred": pred_val,
        "tipo": "Real"
    })

futuro_js = []
for p in predicciones_raw:
    futuro_js.append({
        "label": f"{nombres_meses[p['Mes']]} {p['Anio']}",
        "anio": p['Anio'],
        "mes": p['Mes'],
        "neto": None,
        "pred": round(p['Neto'] / 1e6, 2),
        "tipo": "Predicción"
    })

full_data = historico_js + futuro_js

# HTML Interactivo
html_content = f"""
<!DOCTYPE html><html><head><meta charset='utf-8'><title>DM Dashboard Interactivo</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
    body{{font-family:'Segoe UI',sans-serif;background:#f4f7f6;padding:20px;color:#333}}
    .container{{max-width:1100px;margin:auto}}
    .header{{background:#fff;padding:20px;border-radius:12px;margin-bottom:20px;display:flex;justify-content:space-between;align-items:center;box-shadow:0 2px 5px rgba(0,0,0,0.05)}}
    .logo{{max-height:60px}}
    .filters{{background:#fff;padding:20px;border-radius:12px;margin-bottom:20px;display:grid;grid-template-columns:1fr 1fr;gap:20px;box-shadow:0 2px 5px rgba(0,0,0,0.05)}}
    select{{width:100%;padding:10px;border-radius:8px;border:1px solid #ddd;outline:none;height:100px}}
    .chart-box{{background:#fff;padding:25px;border-radius:12px;box-shadow:0 2px 5px rgba(0,0,0,0.05)}}
    .info{{margin-top:20px;font-size:13px;color:#666;line-height:1.6}}
</style></head><body>
<div class="container">
    <div class="header">
        <img src="logo_dm.png" alt="DM" class="logo">
        <div style="text-align:right"><h1>Dashboard Interactivo</h1><p>Control total de visualización</p></div>
    </div>

    <div class="filters">
        <div>
            <label><b>Filtrar por Año(s):</b> (Ctrl/Cmd + clic para varios)</label>
            <select id="yearFilter" multiple onchange="updateChart()">
                {"".join([f'<option value="{a}" selected>{a}</option>' for a in sorted(mensual_full['Anio'].unique())])}
            </select>
        </div>
        <div>
            <label><b>Filtrar por Mes(es):</b></label>
            <select id="monthFilter" multiple onchange="updateChart()">
                {"".join([f'<option value="{i}" selected>{n}</option>' for i, n in nombres_meses.items()])}
            </select>
        </div>
    </div>
    
    <div class="chart-box"><canvas id="chart"></canvas></div>

    <div class="info">
        <h3>Nota del Modelo</h3>
        <p>Este gráfico es interactivo. Puedes seleccionar combinaciones de años y meses para comparar rendimientos estacionales. La línea naranja representa la predicción basada en la inercia de los últimos meses y el comportamiento histórico del mes seleccionado.</p>
    </div>
</div>

<script>
    const fullData = {full_data};
    let chart;

    function updateChart() {{
        const selectedYears = Array.from(document.getElementById('yearFilter').selectedOptions).map(o => parseInt(o.value));
        const selectedMonths = Array.from(document.getElementById('monthFilter').selectedOptions).map(o => parseInt(o.value));
        
        const filtered = fullData.filter(d => selectedYears.includes(d.anio) && selectedMonths.includes(d.mes));
        
        const labels = filtered.map(d => d.label);
        const netos = filtered.map(d => d.neto);
        const preds = filtered.map(d => d.pred);

        if(chart) chart.destroy();
        
        const ctx = document.getElementById('chart').getContext('2d');
        chart = new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: labels,
                datasets: [
                    {{ label: 'Ventas Reales (M)', data: netos, backgroundColor: '#1a4fa0', borderRadius: 5 }},
                    {{ label: 'Tendencia/Predicción (M)', data: preds, type: 'line', borderColor: '#f28e2b', tension: 0.3, pointRadius: 4 }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{ y: {{ beginAtZero: true, ticks: {{ callback: v => '$' + v + 'M' }} }} }}
            }}
        }});
    }}
    
    updateChart(); // Carga inicial
</script></body></html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)
print("Dashboard interactivo generado.")
