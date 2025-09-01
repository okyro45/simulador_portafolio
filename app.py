import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from io import BytesIO

st.set_page_config(page_title="Simulador Portafolio de Inversiones", layout="wide")

st.title("📊 Simulador de Portafolio de Inversiones (PEN & USD)")

# ===================== SESSION STATE PARA GUARDAR DATOS =====================
if "acciones_data" not in st.session_state:
    st.session_state["acciones_data"] = "EmpresaA,100,5000,50,2,PEN\nEmpresaB,50,3000,60,1.5,USD"

if "bonos_data" not in st.session_state:
    st.session_state["bonos_data"] = "Gobierno,10,2000,24,0.05,PEN\nEmpresaC,20,4000,12,0.06,USD"

if "fondos_data" not in st.session_state:
    st.session_state["fondos_data"] = "FondoA,30,6000,0.07,24,PEN"

if "depositos_data" not in st.session_state:
    st.session_state["depositos_data"] = "BancoA,5000,0.04,12,PEN\nBancoB,3000,0.05,6,USD"

# ===================== INPUTS GENERALES =====================
st.sidebar.header("Parámetros Generales")
tipo_cambio = st.sidebar.number_input("Tipo de cambio PEN/USD", min_value=1.0, value=3.80, step=0.01)
moneda_base = st.sidebar.radio("Moneda base para cálculos", ["PEN", "USD"])
nivel_confianza = st.sidebar.slider("Nivel de confianza (VaR, CVaR)", 90, 99, 95) / 100
volatilidad_mercado = st.sidebar.number_input("Volatilidad del mercado (%)", min_value=0.1, value=15.0) / 100
rend_min_esperado = st.sidebar.number_input("Rendimiento mínimo esperado (%)", min_value=0.0, value=2.0, step=0.1) / 100
tasa_libre_riesgo = st.sidebar.number_input("Tasa libre de riesgo (%)", min_value=0.0, value=2.0, step=0.1) / 100

# ===================== INPUTS DE ACTIVOS =====================
st.subheader("📈 Acciones")
acciones_data = st.text_area(
    "Ingrese datos de Acciones (emisor,num_acciones,importe,precio,dividendo,moneda PEN/USD):",
    st.session_state["acciones_data"],
    key="acciones_data"
)

st.subheader("💵 Bonos")
bonos_data = st.text_area(
    "Ingrese datos de Bonos (emisor,num_bonos,importe,plazo_meses,tasa_anual,moneda PEN/USD):",
    st.session_state["bonos_data"],
    key="bonos_data"
)

st.subheader("🏢 Fondos de Inversión Inmobiliarios")
fondos_data = st.text_area(
    "Ingrese datos de Fondos (emisor,num_unidades,importe,tasa_anual,plazo_meses,moneda PEN/USD):",
    st.session_state["fondos_data"],
    key="fondos_data"
)

st.subheader("🏦 Depósitos a Plazo Fijo")
depositos_data = st.text_area(
    "Ingrese datos de Depósitos (emisor,importe,tasa_anual,plazo_meses,moneda PEN/USD):",
    st.session_state["depositos_data"],
    key="depositos_data"
)

# ===================== FUNCIONES DE PROCESAMIENTO =====================
def procesar_acciones(data):
    rows = [x.split(",") for x in data.strip().split("\n")]
    df = pd.DataFrame(rows, columns=["Emisor","Num_Acciones","Importe","Precio","Dividendo","Moneda"])
    df = df.astype({"Num_Acciones":int,"Importe":float,"Precio":float,"Dividendo":float})
    df["Ganancia"] = df["Num_Acciones"] * df["Dividendo"]
    return df

def procesar_bonos(data):
    rows = [x.split(",") for x in data.strip().split("\n")]
    df = pd.DataFrame(rows, columns=["Emisor","Num_Bonos","Importe","Plazo_Meses","Tasa","Moneda"])
    df = df.astype({"Num_Bonos":int,"Importe":float,"Plazo_Meses":int,"Tasa":float})
    df["Ganancia"] = df["Importe"] * df["Tasa"] * (df["Plazo_Meses"]/1200)
    return df

def procesar_fondos(data):
    rows = [x.split(",") for x in data.strip().split("\n")]
    df = pd.DataFrame(rows, columns=["Emisor","Num_Unidades","Importe","Tasa","Plazo_Meses","Moneda"])
    df = df.astype({"Num_Unidades":int,"Importe":float,"Tasa":float,"Plazo_Meses":int})
    df["Ganancia"] = df["Importe"] * df["Tasa"] * (df["Plazo_Meses"]/1200)
    return df

def procesar_depositos(data):
    rows = [x.split(",") for x in data.strip().split("\n")]
    df = pd.DataFrame(rows, columns=["Emisor","Importe","Tasa","Plazo_Meses","Moneda"])
    df = df.astype({"Importe":float,"Tasa":float,"Plazo_Meses":int})
    df["Ganancia"] = df["Importe"] * df["Tasa"] * (df["Plazo_Meses"]/1200)
    return df

# Conversion mejorada
def convertir_moneda(df, tipo_cambio, moneda_base):
    df = df.copy()
    columnas_convertibles = ["Importe", "Ganancia", "Precio", "Dividendo"]

    if moneda_base == "PEN":
        for col in columnas_convertibles:
            if col in df.columns:
                df.loc[df["Moneda"] == "USD", col] *= tipo_cambio
        df.loc[df["Moneda"] == "USD", "Moneda"] = "PEN"

    elif moneda_base == "USD":
        for col in columnas_convertibles:
            if col in df.columns:
                df.loc[df["Moneda"] == "PEN", col] /= tipo_cambio
        df.loc[df["Moneda"] == "PEN", "Moneda"] = "USD"

    return df

# ===================== PROCESAMIENTO =====================
acciones = procesar_acciones(st.session_state["acciones_data"])
bonos = procesar_bonos(st.session_state["bonos_data"])
fondos = procesar_fondos(st.session_state["fondos_data"])
depositos = procesar_depositos(st.session_state["depositos_data"])

acciones = convertir_moneda(acciones, tipo_cambio, moneda_base)
bonos = convertir_moneda(bonos, tipo_cambio, moneda_base)
fondos = convertir_moneda(fondos, tipo_cambio, moneda_base)
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
    depositos[["Emisor","Importe","Ganancia","Moneda"]]
], axis=0)

total_invertido = portafolio["Importe"].sum()
total_ganancia = portafolio["Ganancia"].sum()
rendimiento_esperado = total_ganancia / total_invertido

np.random.seed(42)
rendimientos = np.random.normal(rendimiento_esperado, volatilidad_mercado, 1000)
std_dev = np.std(rendimientos)
rendimiento_promedio = np.mean(rendimientos)
sharpe_ratio = (rendimiento_promedio - tasa_libre_riesgo) / std_dev

mercado = np.random.normal(0.08, 0.15, 1000)
modelo = LinearRegression().fit(mercado.reshape(-1,1), rendimientos)
beta = modelo.coef_[0]
alpha = modelo.intercept_
r_cuadrado = modelo.score(mercado.reshape(-1,1), rendimientos)

VaR = np.percentile(rendimientos, (1-nivel_confianza)*100)
CVaR = rendimientos[rendimientos <= VaR].mean()
covarianza = np.cov(rendimientos, mercado)[0,1]

# ===================== MOSTRAR =====================
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
    acciones_resumen = acciones.copy(); acciones_resumen["Categoría"] = "Acciones"
    bonos_resumen = bonos.copy(); bonos_resumen["Categoría"] = "Bonos"
    fondos_resumen = fondos.copy(); fondos_resumen["Categoría"] = "Fondos Inmobiliarios"
    depositos_resumen = depositos.copy(); depositos_resumen["Categoría"] = "Depósitos"
    resumen = pd.concat([acciones_resumen, bonos_resumen, fondos_resumen, depositos_resumen], axis=0)
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

# ===================== CARGAR DESDE EXCEL =====================
archivo_excel = st.file_uploader("📂 Cargar simulación desde Excel", type=["xlsx"])
if archivo_excel is not None:
    cargado_acciones = pd.read_excel(archivo_excel, sheet_name="Acciones")
    cargado_bonos = pd.read_excel(archivo_excel, sheet_name="Bonos")
    cargado_fondos = pd.read_excel(archivo_excel, sheet_name="Fondos")
    cargado_depositos = pd.read_excel(archivo_excel, sheet_name="Depositos")

    # Convertir a string para los text_area
    acciones_str = "\n".join(
        cargado_acciones.astype(str).apply(lambda row: ",".join(row), axis=1)
    )
    bonos_str = "\n".join(
        cargado_bonos.astype(str).apply(lambda row: ",".join(row), axis=1)
    )
    fondos_str = "\n".join(
        cargado_fondos.astype(str).apply(lambda row: ",".join(row), axis=1)
    )
    depositos_str = "\n".join(
        cargado_depositos.astype(str).apply(lambda row: ",".join(row), axis=1)
    )

    # Actualizar session_state correctamente
    st.session_state.update({
        "acciones_data": acciones_str,
        "bonos_data": bonos_str,
        "fondos_data": fondos_str,
        "depositos_data": depositos_str
    })

    st.success("✅ Simulación cargada y campos autocompletados. ¡Puedes seguir editando arriba!")
