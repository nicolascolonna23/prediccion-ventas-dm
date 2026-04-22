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
        # PARCHE: Unificar nombres de columnas
        if 'Monto Total' in df.columns: df = df.rename(columns={'Monto Total': 'Total'})
        if 'Valor Neto' in df.columns: df = df.rename(columns={'Valor Neto': 'Neto'})
        return df
    except: return None

print("Descargando datos...")
dfs = [descargar(u) for u in URLS]
df_completo = pd.concat([d for d in dfs if d is not None], ignore_index=True)

# Asegurar que existan las columnas antes de operar
if 'Total' not in df_completo.columns: df_completo['Total'] = 0
if 'Neto' not in df_completo.columns: df_completo['Neto'] = 0

df_completo = df_completo.rename(columns={"Fecha de Emisión": "Fecha"})
df_completo["Neto"] = pd.to_numeric(df_completo["Neto"], errors='coerce').fillna(0)
df_completo["Total"] = pd.to_numeric(df_completo["Total"], errors='coerce').fillna(0)

# Corrección de Netos 0
df_completo.loc[(df_completo["Neto"] == 0) & (df_completo["Total"] > 0), "Neto"] = df_completo["Total"] / 1.21
df_completo["Fecha"] = pd.to_datetime(df_completo["Fecha"], dayfirst=True, errors="coerce")
df_completo = df_completo.dropna(subset=["Fecha"])
df_completo["Anio"] = df_completo["Fecha"].dt.year.astype(int)
df_completo["Mes"]  = df_completo["Fecha"].dt.month.astype(int)

mensual_full = df_completo.groupby(["Anio", "Mes"])["Neto"].sum().reset_index()
mensual_full = mensual_full.sort_values(["Anio", "Mes"]).reset_index(drop=True)

# MODELO
hoy = date.today()
m_entrenamiento = mensual_full[~((mensual_full["Anio"] == hoy.year) & (mensual_full["Mes"] == hoy.month))].copy()
for m in range(1, 12): m_entrenamiento[f"M_{m}"] = (m_entrenamiento["Mes"] == m).astype(int)
m_entrenamiento["L1"] = m_entrenamiento["Neto"].shift(1)
m_entrenamiento["L2"] = m_entrenamiento["Neto"].shift(2)
m_entrenamiento["L3"] = m_entrenamiento["Neto"].shift(3)
m_train = m_entrenamiento.dropna().copy()

cols_x = ["Anio", "L1", "L2", "L3"] + [f"M_{m}" for m in range(1, 12)]
X = m_train[cols_x].values
y = m_train["Neto"].values
X_m, X_s = X.mean(axis=0), X.std(axis=0)
X_s[X_s == 0] = 1
Xn, y_m, y_s = (X - X_m) / X_s, y.mean(), y.std()
yn = (y - y_m) / y_s
coef, _, _, _ = np.linalg.lstsq(np.c_[np.ones(len(Xn)), Xn], yn, rcond=None)

def predecir(a, m, l1, l2, l3):
    dums = [1 if m == i else 0 for i in range(1, 12)]
    xn = (np.array([a, l1, l2, l3] + dums) - X_m) / X_s
    return (coef[0] + np.dot(coef[1:], xn)) * y_s + y_m

# DATA PARA JS
nombres = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}
historico = []
for i, r in mensual_full.iterrows():
    l1, l2, l3 = (mensual_full.loc[i-1,"Neto"], mensual_full.loc[i-2,"Neto"], mensual_full.loc[i-3,"Neto"]) if i > 2 else (0,0,0)
    p_v = round(predecir(r['Anio'], r['Mes'], l1, l2, l3)/1e6, 2) if i > 2 else 0
    historico.append({"lab": f"{nombres[int(r['Mes'])]} {int(r['Anio'])}", "a": int(r['Anio']), "m": int(r['Mes']), "n": round(r['Neto']/1e6, 2), "p": p_v})

u = mensual_full.iloc[-1]
curr_l = [float(u["Neto"]), float(mensual_full.iloc[-2]["Neto"]), float(mensual_full.iloc[-3]["Neto"])]
m_a, a_a = int(u["Mes"]), int(u["Anio"])
futuro = []
for i in range(3):
    m_a += 1
    if m_a > 12: m_a=1; a_a+=1
    p_f = predecir(a_a, m_a, curr_l[0], curr_l[1], curr_l[2])
    futuro.append({"lab": f"{nombres[m_a]} {a_a}", "a": a_a, "m": m_a, "n": None, "p": round(p_f/1e6, 2)})
    curr_l = [p_f, curr_l[0], curr_l[1]]

years_avail = sorted(mensual_full['Anio'].unique().tolist())
cards_html = "".join([f'<div class="card"><div>{f["lab"]}</div><div style="font-size:22px;font-weight:bold;color:#1a4fa0">$ {f["p"]:.1f} M</div></div>' for f in futuro])

html_content = f"""
<!DOCTYPE html><html><head><meta charset='utf-8'><script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
    body{{font-family:sans-serif;background:#f4f7f6;padding:20px}}
    .container{{max-width:1000px;margin:auto;background:white;padding:25px;border-radius:15px;box-shadow:0 4px 10px rgba(0,0,0,0.1)}}
    .cards{{display:flex;gap:15px;margin-bottom:25px}}
    .card{{flex:1;padding:15px;border-radius:10px;border:1px solid #eee;text-align:center;background:#fff;box-shadow:0 2px 4px rgba(0,0,0,0.05)}}
    .filters{{display:flex;gap:20px;margin-bottom:20px;background:#f9f9f9;padding:15px;border-radius:10px}}
    select{{flex:1;height:120px;border:1px solid #ccc;border-radius:5px;padding:5px}}
</style></head><body>
<div class="container">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">
        <img src="logo_dm.png" style="height:50px">
        <h2>Dashboard DM Interactivo</h2>
    </div>
    <div class="cards">{cards_html}</div>
    <div class="filters">
        <div style="flex:1"><b>Años (Ctrl/Cmd+clic):</b><br><select id="yF" multiple onchange="u()"></select></div>
        <div style="flex:1"><b>Meses:</b><br><select id="mF" multiple onchange="u()"></select></div>
    </div>
    <canvas id="chart"></canvas>
    <div style="margin-top:20px; font-size:14px; color:#555; background:#f0f4f8; padding:15px; border-radius:10px;">
        <b>¿Cómo funciona?</b> El modelo analiza la estacionalidad de los últimos 5 años y la inercia (Lags) de los últimos 3 meses para proyectar el futuro.
    </div>
</div>
<script>
    const data = {historico + futuro};
    const ys = {years_avail};
    const ms = {list(nombres.items())};
    let myChart;

    const yS = document.getElementById('yF');
    ys.forEach(y => {{ let o = new Option(y, y); o.selected = true; yS.add(o); }});
    const mS = document.getElementById('mF');
    ms.forEach(([v, n]) => {{ let o = new Option(n, v); o.selected = true; mS.add(o); }});

    function u() {{
        const sy = Array.from(yS.selectedOptions).map(o => parseInt(o.value));
        const sm = Array.from(mS.selectedOptions).map(o => parseInt(o.value));
        const f = data.filter(d => sy.includes(d.a) && sm.includes(d.m));
        if(myChart) myChart.destroy();
        myChart = new Chart(document.getElementById('chart'), {{
            type: 'bar',
            data: {{
                labels: f.map(d => d.lab),
                datasets: [
                    {{ label: 'Ventas Reales (M)', data: f.map(d => d.n), backgroundColor: '#1a4fa0', borderRadius: 5 }},
                    {{ label: 'Tendencia IA (M)', data: f.map(d => d.p), type: 'line', borderColor: '#f28e2b', tension: 0.3 }}
                ]
            }}
        }});
    }}
    u();
</script></body></html>
"""
with open("index.html", "w", encoding="utf-8") as f: f.write(html_content)
