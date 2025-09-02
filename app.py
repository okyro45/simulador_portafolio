import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from io import BytesIO

st.set_page_config(page_title="Simulador Portafolio de Inversiones", layout="wide")
st.title("📊 Simulador de Portafolio de Inversiones (PEN & USD)")

# ===================== DEFAULTS EN SESSION_STATE =====================
defaults = {
    "acciones_input": "EmpresaA,100,5000,50,2,PEN\nEmpresaB,50,3000,60,1.5,USD",
    "bonos_input": "Gobierno,10,2000,24,0.05,PEN\nEmpresaC,20,4000,12,0.06,USD",
    "fondos_input": "FondoA,30,6000,0.07,24,PEN\nFondoB,20,4000,0.08,36,USD",
    "depositos_input": "BancoA,5000,0.04,12,PEN\nBancoB,3000,0.05,6,USD",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ===================== SIDEBAR: PARÁMETROS =====================
st.sidebar.header("Parámetros Generales")
tipo_cambio = st.sidebar.number_input("Tipo de cambio PEN/USD", min_value=1.0, value=3.80, step=0.01)
moneda_base = st.sidebar.radio("Moneda base para cálculos", ["PEN", "USD"])
nivel_confianza = st.sidebar.slider("Nivel de confianza (VaR, CVaR)", 90, 99, 95) / 100
volatilidad_mercado = st.sidebar.number_input("Volatilidad del mercado (%)", min_value=0.1, value=15.0) / 100
rend_min_esperado = st.sidebar.number_input("Rendimiento mínimo esperado (%)", min_value=0.0, value=2.0, step=0.1) / 100
tasa_libre_riesgo = st.sidebar.number_input("Tasa libre de riesgo (%)", min_value=0.0, value=2.0, step=0.1) / 100

# ===================== CARGAR DESDE EXCEL =====================
archivo_excel = st.sidebar.file_uploader("📂 Cargar simulación (Excel con hojas)", type=["xlsx"])

if archivo_excel is not None:
    try:
        cargado_acciones = pd.read_excel(archivo_excel, sheet_name="Acciones")
        cargado_bonos = pd.read_excel(archivo_excel, sheet_name="Bonos")
        cargado_fondos = pd.read_excel(archivo_excel, sheet_name="Fondos")
        cargado_depositos = pd.read_excel(archivo_excel, sheet_name="Depositos")

        # Convertir a string para los text_area
        def df_to_str(df):
            return "\n".join(df.astype(str).apply(lambda row: ",".join(row), axis=1))

        st.session_state["acciones_input"]  = df_to_str(cargado_acciones)
        st.session_state["bonos_input"]     = df_to_str(cargado_bonos)
        st.session_state["fondos_input"]    = df_to_str(cargado_fondos)
        st.session_state["depositos_input"] = df_to_str(cargado_depositos)

        st.sidebar.success("✅ Simulación cargada y campos autocompletados.")
    except Exception as e:
        st.sidebar.error(f"❌ Error al leer el Excel: {e}")

# ===================== INPUTS DE ACTIVOS =====================
st.subheader("📈 Acciones")
st.text_area(
    "Ingrese datos de Acciones (emisor,num_acciones,importe,precio,dividendo,moneda PEN/USD):",
    st.session_state["acciones_input"],
    key="acciones_input"
)

st.subheader("💵 Bonos")
st.text_area(
    "Ingrese datos de Bonos (emisor,num_bonos,importe,plazo_meses,tasa_anual,moneda PEN/USD):",
    st.session_state["bonos_input"],
    key="bonos_input"
)

st.subheader("🏢 Fondos de Inversión Inmobiliarios")
st.text_area(
    "Ingrese datos de Fondos (emisor,num_unidades,importe,tasa_anual,plazo_meses,moneda PEN/USD):",
    st.session_state["fondos_input"],
    key="fondos_input"
)

st.subheader("🏦 Depósitos a Plazo Fijo")
st.text_area(
    "Ingrese datos de Depósitos (emisor,importe,tasa_anual,plazo_meses,moneda PEN/USD):",
    st.session_state["depositos_input"],
    key="depositos_input"
)

# ===================== FUNCIONES DE PROCESAMIENTO =====================
def procesar_acciones(data):
    rows = []
    for x in data.strip().split("\n"):
        if not x.strip():
            continue
        partes = [p.strip() for p in x.split(",")]
        if len(partes) != 6:
            continue  # ignorar filas mal formadas
        rows.append(partes)

    if not rows:
        return pd.DataFrame(columns=["Emisor","Num_Acciones","Importe","Precio","Dividendo","Moneda"])

    df = pd.DataFrame(rows, columns=["Emisor","Num_Acciones","Importe","Precio","Dividendo","Moneda"])
    df = df.astype({"Num_Acciones":int,"Importe":float,"Precio":float,"Dividendo":float})
    df["Ganancia"] = df["Num_Acciones"] * df["Dividendo"]
    return df

def procesar_bonos(data):
    rows = [x.split(",") for x in data.strip().split("\n") if x.strip()]
    df = pd.DataFrame(rows, columns=["Emisor","Num_Bonos","Importe","Plazo_Meses","Tasa","Moneda"])
    df = df.astype({"Num_Bonos":int,"Importe":float,"Plazo_Meses":int,"Tasa":float})
    df["Ganancia"] = df["Importe"] * df["Tasa"] * (df["Plazo_Meses"]/1200)
    return df

def procesar_fondos(data):
    rows = [x.split(",") for x in data.strip().split("\n") if x.strip()]
    df = pd.DataFrame(rows, columns=["Emisor","Num_Unidades","Importe","Tasa","Plazo_Meses","Moneda"])
    df = df.astype({"Num_Unidades":int,"Importe":float,"Tasa":float,"Plazo_Meses":int})
    df["Ganancia"] = df["Importe"] * df["Tasa"] * (df["Plazo_Meses"]/1200)
    return df

def procesar_depositos(data):
    rows = [x.split(",") for x in data.strip().split("\n") if x.strip()]
    df = pd.DataFrame(rows, columns=["Emisor","Importe","Tasa","Plazo_Meses","Moneda"])
    df = df.astype({"Importe":float,"Tasa":float,"Plazo_Meses":int})
    df["Ganancia"] = df["Importe"] * df["Tasa"] * (df["Plazo_Meses"]/1200)
    return df

def convertir_moneda(df, tipo_cambio, moneda_base):
    df = df.copy()
    columnas_convertibles = ["Importe", "Ganancia", "Precio", "Dividendo"]
    if moneda_base == "PEN":
        for col in columnas_convertibles:
            if col in df.columns:
                df.loc[df["Moneda"] == "USD", col] *= tipo_cambio
        df.loc[df["Moneda"] == "USD", "Moneda"] = "PEN"
    else:  # USD
        for col in columnas_convertibles:
            if col in df.columns:
                df.loc[df["Moneda"] == "PEN", col] /= tipo_cambio
        df.loc[df["Moneda"] == "PEN", "Moneda"] = "USD"
    return df

# ===================== PROCESAMIENTO =====================
acciones  = procesar_acciones(st.session_state["acciones_input"])
bonos     = procesar_bonos(st.session_state["bonos_input"])
fondos    = procesar_fondos(st.session_state["fondos_input"])
depositos = procesar_depositos(st.session_state["depositos_input"])

acciones  = convertir_moneda(acciones,  tipo_cambio, moneda_base)
bonos     = convertir_moneda(bonos,     tipo_cambio, moneda_base)
fondos    = convertir_moneda(fondos,    tipo_cambio, moneda_base)
depositos = convertir_moneda(depositos, tipo_cambio, moneda_base)

# ===================== RESULTADOS POR ACTIVO =====================
st.subheader("📊 Resultados por Categoría")
st.write("Acciones", acciones)
st.write("Bonos", bonos)
st.write("Fondos de Inversión Inmobiliarios", fondos)
st.write("Depósitos a Plazo", depositos)

# ===================== MÉTRICAS =====================
st.header("📈 Métricas del Portafolio")
portafolio = pd.concat([
    acciones[["Emisor","Importe","Ganancia","Moneda"]],
    bonos[["Emisor","Importe","Ganancia","Moneda"]],
    fondos[["Emisor","Importe","Ganancia","Moneda"]],
    depositos[["Emisor","Importe","Ganancia","Moneda"]],
], axis=0)

total_invertido = portafolio["Importe"].sum()
total_ganancia = portafolio["Ganancia"].sum()
rendimiento_esperado = total_ganancia / total_invertido if total_invertido else 0.0

np.random.seed(42)
rendimientos = np.random.normal(rendimiento_esperado, volatilidad_mercado or 1e-9, 1000)
std_dev = float(np.std(rendimientos)) or 1e-9
rendimiento_promedio = float(np.mean(rendimientos))
sharpe_ratio = (rendimiento_promedio - tasa_libre_riesgo) / std_dev

mercado = np.random.normal(0.08, 0.15, 1000)
modelo = LinearRegression().fit(mercado.reshape(-1,1), rendimientos)
beta = float(modelo.coef_[0])
alpha = float(modelo.intercept_)
r_cuadrado = float(modelo.score(mercado.reshape(-1,1), rendimientos))

VaR = float(np.percentile(rendimientos, (1-nivel_confianza)*100))
CVaR = float(rendimientos[rendimientos <= VaR].mean())
covarianza = float(np.cov(rendimientos, mercado)[0,1])

col1, col2 = st.columns(2)
with col1:
    st.metric("💰 Total Invertido", f"{total_invertido:,.2f}")
    st.metric("📈 Ganancia Total", f"{total_ganancia:,.2f}")
    st.metric("🎯 Rendimiento Esperado", f"{rendimiento_esperado:.2%}")
    st.metric("📉 Desviación Estándar", f"{std_dev:.2%}")
    st.metric("📊 Ratio de Sharpe", f"{sharpe_ratio:.2f}")
with col2:
    st.metric("📈 Beta", f"{beta:.2f}")
    st.metric("✨ Alfa", f"{alpha:.2%}")
    st.metric("🔎 R²", f"{r_cuadrado:.2f}")
    st.metric("⚠️ VaR", f"{VaR:.2%}")
    st.metric("🔥 CVaR", f"{CVaR:.2%}")
    st.metric("🔗 Covarianza", f"{covarianza:.4f}")

# ===================== EXPORTAR EXCEL =====================
def exportar_excel(acciones, bonos, fondos, depositos, metrics):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        acciones.to_excel(writer, index=False, sheet_name="Acciones")
        bonos.to_excel(writer, index=False, sheet_name="Bonos")
        fondos.to_excel(writer, index=False, sheet_name="Fondos")
        depositos.to_excel(writer, index=False, sheet_name="Depositos")
        pd.DataFrame([metrics]).to_excel(writer, index=False, sheet_name="Métricas")
    return output.getvalue()

metrics = {
    "Total Invertido": total_invertido,
    "Ganancia Total": total_ganancia,
    "Rendimiento Esperado": rendimiento_esperado,
    "Desviación Estándar": std_dev,
    "Sharpe Ratio": sharpe_ratio,
    "Beta": beta,
    "Alfa": alpha,
    "R²": r_cuadrado,
    "VaR": VaR,
    "CVaR": CVaR,
    "Covarianza": covarianza
}

excel_bytes = exportar_excel(acciones, bonos, fondos, depositos, metrics)
st.download_button(
    label="⬇️ Descargar Simulación en Excel",
    data=excel_bytes,
    file_name="simulacion_portafolio.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

def exportar_excel_resumen(acciones, bonos, fondos, depositos):
    output = BytesIO()
    a = acciones.copy();  a["Categoría"] = "Acciones"
    b = bonos.copy();     b["Categoría"] = "Bonos"
    f = fondos.copy();    f["Categoría"] = "Fondos Inmobiliarios"
    d = depositos.copy(); d["Categoría"] = "Depósitos"
    resumen = pd.concat([a, b, f, d], axis=0)
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        resumen.to_excel(writer, index=False, sheet_name="Resumen")
    return output.getvalue()

excel_resumen = exportar_excel_resumen(acciones, bonos, fondos, depositos)
st.download_button(
    label="⬇️ Descargar Resumen Consolidado en Excel",
    data=excel_resumen,
    file_name="simulacion_resumen.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


