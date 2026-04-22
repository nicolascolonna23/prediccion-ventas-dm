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
        return df
    except Exception as e:
        print(f"Error descargando {url}: {e}")
        return None

print("Descargando datos...")
dfs = [descargar(u) for u in URLS]
dfs = [d for d in dfs if d is not None and len(d) > 100]
df = pd.concat(dfs, ignore_index=True)
print(f"Total filas: {len(df)}")

df = df.rename(columns={"Fecha de Emisión": "Fecha"})
df = df[["Fecha", "Neto", "Id Cliente"]].dropna(subset=["Fecha", "Neto"])
df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["Fecha"])
df["Anio"] = df["Fecha"].dt.year
df["Mes"]  = df["Fecha"].dt.month
df = df[df["Anio"] >= 2021]

hoy = date.today()
conteo_actual = df[(df["Anio"] == hoy.year) & (df["Mes"] == hoy.month)].shape[0]
mes_ant = hoy.month - 1 if hoy.month > 1 else 12
anio_ant = hoy.year if hoy.month > 1 else hoy.year - 1
conteo_ant = df[(df["Anio"] == anio_ant) & (df["Mes"] == mes_ant)].shape[0]
if conteo_actual < conteo_ant * 0.85:
    print(f"Excluyendo {hoy.month}/{hoy.year} por mes incompleto")
    df = df[~((df["Anio"] == hoy.year) & (df["Mes"] == hoy.month))]

print("Descargando IPC del INDEC...")
ipc_ok = False
ipc_df = None
try:
    url_indec = "https://apis.datos.gob.ar/series/api/series/?ids=148.3_INIVELNAL_DICI_M_26&limit=200&format=json"
    resp = requests.get(url_indec, timeout=30).json()
    ipc_df = pd.DataFrame(resp["data"], columns=["fecha", "ipc"])
    ipc_df["fecha"] = pd.to_datetime(ipc_df["fecha"])
    ipc_df["Anio"] = ipc_df["fecha"].dt.year
    ipc_df["Mes"]  = ipc_df["fecha"].dt.month
    ipc_df = ipc_df[["Anio", "Mes", "ipc"]].dropna()
    base = ipc_df[(ipc_df["Anio"] == 2024) & (ipc_df["Mes"] == 12)]["ipc"].values
    ipc_df["deflactor"] = base[0] / ipc_df["ipc"] if len(base) > 0 else 1.0
    ipc_ok = True
    print("IPC OK")
except Exception as e:
    print(f"Error IPC: {e}")

mensual = df.groupby(["Anio", "Mes"])["Neto"].sum().reset_index()
mensual = mensual.sort_values(["Anio", "Mes"]).reset_index(drop=True)

if ipc_ok:
    mensual = mensual.merge(ipc_df[["Anio", "Mes", "deflactor"]], on=["Anio", "Mes"], how="left")
    mensual["deflactor"] = mensual["deflactor"].fillna(1.0)
else:
    mensual["deflactor"] = 1.0
mensual["Neto_real"] = mensual["Neto"] * mensual["deflactor"]

mensual["Lag1"] = mensual["Neto_real"].shift(1)
mensual["Lag2"] = mensual["Neto_real"].shift(2)
mensual["Lag3"] = mensual["Neto_real"].shift(3)
mensual = mensual.dropna().reset_index(drop=True)

X = mensual[["Anio", "Mes", "Lag1", "Lag2", "Lag3"]].values
y = mensual["Neto_real"].values
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X_std[X_std == 0] = 1
Xn = (X - X_mean) / X_std
y_mean, y_std = y.mean(), y.std()
yn = (y - y_mean) / y_std
A = np.c_[np.ones(len(Xn)), Xn]
coef, _, _, _ = np.linalg.lstsq(A, yn, rcond=None)

def predecir_real(anio, mes, lag1, lag2, lag3):
    x = np.array([anio, mes, lag1, lag2, lag3])
    xn = (x - X_mean) / X_std
    return (coef[0] + np.dot(coef[1:], xn)) * y_std + y_mean

def real_a_nominal(valor_real, anio, mes):
    if not ipc_ok:
        return valor_real
    fila = ipc_df[(ipc_df["Anio"] == anio) & (ipc_df["Mes"] == mes)]
    if len(fila) == 0:
        fila = ipc_df.tail(1)
    return valor_real / fila["deflactor"].values[0]

ultimo = mensual.iloc[-1]
anio_ult = int(ultimo["Anio"])
mes_ult  = int(ultimo["Mes"])

meses_sig = []
for i in range(1, 4):
    m = mes_ult + i
    a = anio_ult
    if m > 12:
        m -= 12
        a += 1
    meses_sig.append((a, m))

lags = [float(mensual.iloc[-1]["Neto_real"]),
        float(mensual.iloc[-2]["Neto_real"]),
        float(mensual.iloc[-3]["Neto_real"])]
predicciones = []
for a, m in meses_sig:
    pr = predecir_real(a, m, lags[0], lags[1], lags[2])
    pn = real_a_nominal(pr, a, m)
    predicciones.append((a, m, pr, pn))
    lags = [pr, lags[0], lags[1]]

nombres = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",
           6:"Jun",7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}

print("\n=== PREDICCION ===")
for a, m, pr, pn in predicciones:
    print(f"{nombres[m]} {a}: $ {pr/1e6:,.1f} M reales | $ {pn/1e6:,.1f} M nominales")

top10 = (df.groupby("Id Cliente")["Neto"]
           .sum().sort_values(ascending=False)
           .head(10).reset_index())
top10.columns = ["Cliente", "Neto Total"]
top_max = top10["Neto Total"].max()

if os.path.exists("logo_dm.png"):
    with open("logo_dm.png", "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    logo_tag = '<img src="data:image/png;base64,' + logo_b64 + '" style="height:48px;">'
else:
    logo_tag = '<span style="font-size:18px;font-weight:bold;color:#1a4fa0;">DM Vencemos Distancias</span>'

etiq_hist  = [f"{nombres[int(r.Mes)]} {int(r.Anio)}" for _, r in mensual.iterrows()]
netos_hist = [round(r.Neto_real / 1e6, 1) for _, r in mensual.iterrows()]
preds_hist = [round(predecir_real(r.Anio, r.Mes, r.Lag1, r.Lag2, r.Lag3) / 1e6, 1) for _, r in mensual.iterrows()]
etiq_fut   = [f"{nombres[m]} {a}" for a, m, _, _ in predicciones]
preds_fut  = [round(pr / 1e6, 1) for _, _, pr, _ in predicciones]

etiquetas  = etiq_hist + etiq_fut
netos_js   = "[" + ",".join(str(v) for v in netos_hist) + "," + ",".join("null" for _ in predicciones) + "]"
preds_js   = "[" + ",".join(str(v) for v in preds_hist) + "," + ",".join(str(v) for v in preds_fut) + "]"
n_hist     = len(etiq_hist)
fecha_txt  = hoy.strftime("%d/%m/%Y")

cards_html = ""
for a, m, pr, pn in predicciones:
    cards_html += (
        '<div class="card">'
        '<div class="card-label">' + nombres[m] + " " + str(a) + "</div>"
        '<div class="card-value">$ ' + f"{pn/1e6:,.0f}" + " M</div>"
        '<div class="card-sub">$ ' + f"{pr/1e6:,.0f}" + " M pesos dic-2024</div>"
        "</div>"
    )

top10_rows = ""
for _, r in top10.iterrows():
    pct = round(r["Neto Total"] / top_max * 100)
    top10_rows += (
        "<tr>"
        '<td style="padding:10px 14px;font-weight:500;color:#1a4fa0;">' + str(int(r["Cliente"])) + "</td>"
        '<td style="padding:10px 14px;">'
        '<div style="background:#e8eef7;border-radius:4px;height:10px;width:100%;">'
        '<div style="background:#1a4fa0;border-radius:4px;height:10px;width:' + str(pct) + '%;"></div>'
        "</div></td>"
        '<td style="padding:10px 14px;text-align:right;font-weight:500;">$ ' + f"{r['Neto Total']/1e6:,.1f}" + " M</td>"
        "</tr>"
    )

labels_js = str(etiquetas)

html_parts = []
html_parts.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
html_parts.append("<title>Prediccion de Ventas - DM</title>")
html_parts.append('<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>')
html_parts.append("""<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Arial,sans-serif;background:#f0f2f5;padding:24px}
.container{max-width:980px;margin:auto}
.header{display:flex;align-items:center;justify-content:space-between;background:white;border-radius:12px;padding:20px 28px;margin-bottom:20px;border-bottom:3px solid #1a4fa0}
.header-right{text-align:right}
.header-right h1{font-size:18px;color:#1a4fa0;font-weight:bold}
.header-right p{font-size:12px;color:#888;margin-top:2px}
.cards{display:flex;gap:16px;margin-bottom:20px}
.card{flex:1;background:white;border-radius:12px;padding:20px;text-align:center;border-top:4px solid #f28e2b}
.card-label{font-size:12px;color:#888;margin-bottom:6px;text-transform:uppercase}
.card-value{font-size:26px;font-weight:bold;color:#1a4fa0}
.card-sub{font-size:11px;color:#aaa;margin-top:4px}
.box{background:white;border-radius:12px;padding:24px;margin-bottom:20px}
.box-title{font-size:13px;color:#888;margin-bottom:16px}
.note{font-size:11px;color:#aaa;margin-top:8px}
table{width:100%;border-collapse:collapse}
thead th{font-size:12px;color:#888;text-transform:uppercase;padding:8px 14px;text-align:left;border-bottom:1px solid #eee}
tbody tr:hover{background:#f8f9fb}
.footer{text-align:center;font-size:11px;color:#bbb;margin-top:16px}
</style></head><body><div class="container">""")

html_parts.append('<div class="header">' + logo_tag)
html_parts.append('<div class="header-right"><h1>Prediccion de Ventas</h1>')
html_parts.append("<p>Actualizado: " + fecha_txt + " — Historico desde 2021 · Pesos constantes dic-2024</p>")
html_parts.append("</div></div>")
html_parts.append('<div class="cards">' + cards_html + "</div>")
html_parts.append('<div class="box"><div class="box-title">Ventas netas mensuales vs prediccion (pesos constantes dic-2024, en millones)</div>')
html_parts.append('<canvas id="chart" style="max-height:380px;"></canvas>')
html_parts.append('<div class="note">Valores deflactados por IPC INDEC. Linea naranja: ajuste del modelo y proyeccion futura.</div></div>')
html_parts.append('<div class="box"><div class="box-title">Top 10 clientes por ventas acumuladas (nominales)</div>')
html_parts.append("<table><thead><tr><th>ID Cliente</th><th>Participacion</th>")
html_parts.append('<th style="text-align:right;">Total facturado</th></tr></thead>')
html_parts.append("<tbody>" + top10_rows + "</tbody></table></div>")
html_parts.append('<div class="footer">Generado automaticamente todos los dias · DM Vencemos Distancias</div>')
html_parts.append("</div>")

html_parts.append("<script>")
html_parts.append("const labels=" + labels_js + ";")
html_parts.append("const netos=" + netos_js + ";")
html_parts.append("const predics=" + preds_js + ";")
html_parts.append("const nHist=" + str(n_hist) + ";")
html_parts.append("""
new Chart(document.getElementById('chart'),{
  type:'bar',
  data:{
    labels,
    datasets:[
      {label:'Ventas reales deflactadas (M)',data:netos,backgroundColor:'#1a4fa0',borderRadius:4,order:2},
      {label:'Prediccion (M)',data:predics,type:'line',borderColor:'#f28e2b',
       backgroundColor:labels.map((_,i)=>i>=nHist?'#f28e2b':'transparent'),
       pointRadius:labels.map((_,i)=>i>=nHist?8:3),borderWidth:2.5,tension:0.3,order:1}
    ]
  },
  options:{
    responsive:true,
    plugins:{
      legend:{position:'top'},
      tooltip:{callbacks:{label:ctx=>ctx.parsed.y!==null?'$ '+ctx.parsed.y.toLocaleString('es-AR',{minimumFractionDigits:1})+' M':''}}
    },
    scales:{
      y:{ticks:{callback:val=>'$ '+val.toLocaleString('es-AR')+' M'},grid:{color:'#f0f0f0'}},
      x:{grid:{display:false},ticks:{maxRotation:45}}
    }
  }
});
""")
html_parts.append("</script></body></html>")

with open("index.html", "w", encoding="utf-8") as f:
    f.write("".join(html_parts))
print("index.html generado.")
