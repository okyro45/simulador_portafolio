import streamlit as st
import numpy as np
import pandas as pd

# =============================
# Configuración general
# =============================
st.set_page_config(page_title="Simulador Portafolio (PEN & USD)", layout="wide")
st.title("📈 Simulador de Portafolio de Inversiones — PEN & USD")

st.markdown(
    """
    Esta app te permite **registrar un portafolio** con cuatro clases de activos y **simular métricas de riesgo/retorno**:
    - **Acciones** (importe, precio/acción, dividendo/acción)
    - **Bonos** (importe, plazo [meses], tasa anual, ganancia generada)
    - **Fondos de Inversión Inmobiliarios** (importe, tasa anual, plazo [meses], ganancia generada)
    - **Depósitos a Plazo Fijo** (importe, tasa anual, plazo [meses], ganancia generada)

    Se calculan: **rendimiento esperado, desviación estándar, Sharpe, beta, alfa, R², VaR y CVaR** del portafolio.

    **Notas**
    - Todo se puede ingresar en **PEN o USD**; se convierte a una **moneda base** que defines en la barra lateral.
    - Las métricas de riesgo se calculan con un **modelo paramétrico** (multinormal) usando tus supuestos de retorno y volatilidad.
    - Beta/Alfa/R² se calculan vs. un **benchmark** (también paramétrico) que configuras.
    """
)

# =============================
# Sidebar — Parámetros Globales
# =============================
st.sidebar.header("⚙️ Parámetros Globales")
moneda_base = st.sidebar.selectbox("Moneda base del portafolio", options=["PEN", "USD"], index=0)

tipo_cambio = st.sidebar.number_input(
    "Tipo de cambio USD→PEN (para convertir USD a PEN)",
    min_value=0.0001, value=3.80, step=0.01, format="%.4f",
)

if moneda_base == "USD":
    st.sidebar.info("Si la moneda base es USD, el importe en PEN se divide por el tipo de cambio para convertir a USD.")

st.sidebar.subheader("Supuestos de simulación")
num_paths = st.sidebar.number_input("N° simulaciones (Monte Carlo)", min_value=1000, value=10000, step=1000)
conf = st.sidebar.slider("Nivel de confianza para VaR/CVaR", min_value=0.80, max_value=0.995, value=0.95)
r_f = st.sidebar.number_input("Tasa libre de riesgo anual (para Sharpe)", min_value=-100.0, max_value=100.0, value=2.0, step=0.1, format="%.2f") / 100.0

st.sidebar.subheader("Benchmark (para Beta/Alfa/R²)")
bench_mu = st.sidebar.number_input("Retorno esperado anual benchmark (%)", value=8.0, step=0.5, format="%.2f") / 100.0
bench_sigma = st.sidebar.number_input("Volatilidad anual benchmark (%)", value=15.0, step=0.5, format="%.2f") / 100.0
rho_port_bench = st.sidebar.slider("Correlación Portafolio ↔ Benchmark", min_value=-1.0, max_value=1.0, value=0.60, step=0.05)

st.sidebar.subheader("Correlación entre clases de activos")
rho_assets = st.sidebar.slider("Correlación promedio entre clases (off-diagonal)", min_value=-1.0, max_value=1.0, value=0.25, step=0.05)

# =============================
# Helpers de moneda y cálculos
# =============================

def convertir_a_base(monto: float, moneda: str) -> float:
    """Convierte monto a la moneda base usando el tipo de cambio provisto."""
    if moneda == moneda_base:
        return monto
    if moneda == "USD" and moneda_base == "PEN":
        return monto * tipo_cambio
    if moneda == "PEN" and moneda_base == "USD":
        return monto / tipo_cambio
    return monto


def anualizar_meses(tasa_anual: float, meses: float) -> float:
    """Convierte una tasa anual a rendimiento para un plazo en meses (capitalización simple)."""
    return tasa_anual * (meses / 12.0)


def generar_cov_from_sigmas(sigmas, rho):
    n = len(sigmas)
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                cov[i, j] = sigmas[i] ** 2
            else:
                cov[i, j] = rho * sigmas[i] * sigmas[j]
    return cov


def calcular_var_cvar(returns: np.ndarray, alpha: float):
    """VaR y CVaR (Expected Shortfall) paramétricos/empíricos sobre la serie de retornos simulados.
    returns: array de retornos del portafolio (simulados)
    alpha: nivel de confianza (ej. 0.95)
    Devuelve pérdidas (números positivos) sobre el valor 1 (i.e., sobre la inversión base).
    """
    # Ordenamos de peor a mejor
    sorted_r = np.sort(returns)
    q = np.quantile(sorted_r, 1 - alpha)  # cuantil de pérdidas
    var = -q  # VaR como pérdida positiva
    # CVaR = promedio de cola (retornos <= q)
    tail = sorted_r[sorted_r <= q]
    cvar = -tail.mean() if tail.size > 0 else var
    return var, cvar


# =============================
# DataFrames de entrada por clase de activo
# =============================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Acciones")
    acciones_df = st.data_editor(
        pd.DataFrame(
            {
                "moneda": ["PEN"],
                "importe": [10000.0],
                "precio_accion": [10.0],
                "dividendo_accion": [0.50],
                "ret_esp_anual_%": [10.0],  # retorno total esperado anual (precio + dividendos)
                "vol_anual_%": [25.0],
                "plazo_meses": [12],
            }
        ),
        num_rows="dynamic",
        use_container_width=True,
        help="Puedes añadir o borrar filas. El retorno esperado anual incluye crecimiento de precio y dividendos.",
    )

with col2:
    st.subheader("💵 Bonos")
    bonos_df = st.data_editor(
        pd.DataFrame(
            {
                "moneda": ["PEN"],
                "importe": [10000.0],
                "tasa_anual_%": [8.0],
                "plazo_meses": [12],
                "vol_anual_%": [5.0],
            }
        ),
        num_rows="dynamic",
        use_container_width=True,
    )

col3, col4 = st.columns(2)
with col3:
    st.subheader("🏢 Fondos de Inversión Inmobiliarios (FII)")
    fii_df = st.data_editor(
        pd.DataFrame(
            {
                "moneda": ["USD"],
                "importe": [3000.0],
                "tasa_anual_%": [7.0],
                "plazo_meses": [12],
                "vol_anual_%": [10.0],
            }
        ),
        num_rows="dynamic",
        use_container_width=True,
    )

with col4:
    st.subheader("🏦 Depósitos a Plazo Fijo")
    dpf_df = st.data_editor(
        pd.DataFrame(
            {
                "moneda": ["USD"],
                "importe": [5000.0],
                "tasa_anual_%": [6.0],
                "plazo_meses": [12],
                "vol_anual_%": [0.1],  # típicamente muy baja
            }
        ),
        num_rows="dynamic",
        use_container_width=True,
    )

# =============================
# Construcción del portafolio y métricas por clase
# =============================

def preparar_componentes(acciones_df, bonos_df, fii_df, dpf_df):
    componentes = []  # lista de dicts con info por "activo/clase-item"

    # Acciones: retorno esperado proviene de ret_esp_anual_%
    for _, r in acciones_df.fillna(0).iterrows():
        monto_base = convertir_a_base(float(r["importe"]), str(r["moneda"]))
        precio = float(r["precio_accion"]) if r["precio_accion"] else 0.0
        dividendo = float(r["dividendo_accion"]) if r["dividendo_accion"] else 0.0
        ret_mu = float(r["ret_esp_anual_%"]) / 100.0
        sigma = max(1e-6, float(r["vol_anual_%"]) / 100.0)
        plazo = float(r["plazo_meses"]) if r["plazo_meses"] else 12.0

        shares = (monto_base / precio) if precio > 0 else 0.0
        ganancia = monto_base * anualizar_meses(ret_mu, plazo)  # incluye dividendos implícitos
        ingresos_div = shares * dividendo * (plazo / 12.0)

        componentes.append(
            {
                "clase": "Acción",
                "monto_base": monto_base,
                "ret_mu_anual": ret_mu,
                "sigma_anual": sigma,
                "plazo_meses": plazo,
                "ganancia_generada": ganancia,
                "detalle": {
                    "precio": precio,
                    "dividendo_accion": dividendo,
                    "acciones_estimadas": shares,
                    "ingresos_dividendos": ingresos_div,
                },
            }
        )

    # Bonos / FII / DPF: retorno esperado = tasa_anual_%
    def agregar_clase(df, nombre):
        for _, r in df.fillna(0).iterrows():
            monto_base = convertir_a_base(float(r["importe"]), str(r["moneda"]))
            tasa = float(r["tasa_anual_%"]) / 100.0
            sigma = max(1e-6, float(r["vol_anual_%"]) / 100.0)
            plazo = float(r["plazo_meses"]) if r["plazo_meses"] else 12.0
            ganancia = monto_base * anualizar_meses(tasa, plazo)
            componentes.append(
                {
                    "clase": nombre,
                    "monto_base": monto_base,
                    "ret_mu_anual": tasa,
                    "sigma_anual": sigma,
                    "plazo_meses": plazo,
                    "ganancia_generada": ganancia,
                    "detalle": {},
                }
            )

    agregar_clase(bonos_df, "Bono")
    agregar_clase(fii_df, "FII")
    agregar_clase(dpf_df, "DPF")
    return componentes


componentes = preparar_componentes(acciones_df, bonos_df, fii_df, dpf_df)

if len(componentes) == 0:
    st.warning("Añade al menos un instrumento en alguna tabla para continuar.")
    st.stop()

# Tabla resumen por item
resumen_items = []
for c in componentes:
    resumen_items.append(
        {
            "clase": c["clase"],
            "monto_base": c["monto_base"],
            "ret_esp_anual_%": 100 * c["ret_mu_anual"],
            "vol_anual_%": 100 * c["sigma_anual"],
            "plazo_meses": c["plazo_meses"],
            "ganancia_generada": c["ganancia_generada"],
        }
    )
resumen_df = pd.DataFrame(resumen_items)

st.subheader("📋 Resumen por instrumento (convertido a moneda base)")
st.dataframe(resumen_df, use_container_width=True)

# Totales por moneda de origen

def totales_por_moneda():
    tot_pen, tot_usd = 0.0, 0.0
    for df in [acciones_df, bonos_df, fii_df, dpf_df]:
        for _, r in df.fillna(0).iterrows():
            mon = str(r["moneda"]).upper()
            amt = float(r["importe"]) if r["importe"] else 0.0
            if mon == "PEN":
                tot_pen += amt
            elif mon == "USD":
                tot_usd += amt
    return tot_pen, tot_usd

pen_o, usd_o = totales_por_moneda()

st.info(
    f"**Totales ingresados por moneda de origen** → PEN: {pen_o:,.2f} | USD: {usd_o:,.2f} | "
    f"**Total en base ({moneda_base})**: {sum(c['monto_base'] for c in componentes):,.2f}"
)

# =============================
# Construcción de pesos, medias y covarianzas
# =============================
valores = np.array([c["monto_base"] for c in componentes], dtype=float)
W = valores / valores.sum()
mu = np.array([c["ret_mu_anual"] for c in componentes], dtype=float)
sigmas = np.array([c["sigma_anual"] for c in componentes], dtype=float)

# Matriz de covarianza (misma correlación off-diagonal para simplicidad)
Cov = generar_cov_from_sigmas(sigmas, rho_assets)

# =============================
# Simulación Monte Carlo (anual)
# =============================
np.random.seed(42)
try:
    L = np.linalg.cholesky(Cov)
except np.linalg.LinAlgError:
    # Asegurar semidefinida positiva incrementando diagonal
    jitter = 1e-8
    Cov += np.eye(len(sigmas)) * jitter
    L = np.linalg.cholesky(Cov)

Z = np.random.normal(size=(num_paths, len(sigmas)))
asset_rets = mu + Z.dot(L.T)  # simulaciones anuales por activo

# Retorno del portafolio por simulación
port_rets = asset_rets.dot(W)

# Simular benchmark correlacionado con el portafolio
# Método: R_b = mu_b + sigma_b * (rho * Zp + sqrt(1-rho^2) * eps)
Zp = (port_rets - port_rets.mean()) / (port_rets.std() + 1e-12)
eps = np.random.normal(size=num_paths)
bench_rets = bench_mu + bench_sigma * (rho_port_bench * Zp + np.sqrt(max(0.0, 1 - rho_port_bench**2)) * eps)

# =============================
# Métricas del portafolio
# =============================
port_mu = float(port_rets.mean())
port_sigma = float(port_rets.std(ddof=1))
sharpe = (port_mu - r_f) / (port_sigma + 1e-12)

# Beta/Alfa/R² vs. benchmark
cov_pb = float(np.cov(port_rets, bench_rets, ddof=1)[0, 1])
var_b = float(np.var(bench_rets, ddof=1))
beta = cov_pb / (var_b + 1e-12)
alpha = port_mu - beta * bench_mu
corr_pb = float(np.corrcoef(port_rets, bench_rets)[0, 1])
r2 = corr_pb ** 2

# VaR / CVaR sobre 1 unidad de patrimonio en moneda base
var_loss, cvar_loss = calcular_var_cvar(port_rets, conf)

# Covarianza global del portafolio (escala): W^T Cov W
port_var_theoretical = float(W @ Cov @ W)
port_sigma_theoretical = np.sqrt(port_var_theoretical)

# =============================
# Salidas
# =============================
st.subheader("📐 Métricas del Portafolio (anualizadas)")
met_cols = st.columns(3)
with met_cols[0]:
    st.metric("Rendimiento esperado (μ)", f"{100*port_mu:.2f}%")
    st.metric("Desviación estándar (σ)", f"{100*port_sigma:.2f}%")
    st.metric("Sharpe", f"{sharpe:.2f}")
with met_cols[1]:
    st.metric("Beta (vs. benchmark)", f"{beta:.2f}")
    st.metric("Alfa (anual)", f"{100*alpha:.2f}%")
    st.metric("R²", f"{r2:.2f}")
with met_cols[2]:
    st.metric(f"VaR {int(conf*100)}% (pérdida)", f"{100*var_loss:.2f}%")
    st.metric(f"CVaR {int(conf*100)}% (pérdida)", f"{100*cvar_loss:.2f}%")
    st.metric("σ teórica (WᵀΣW)", f"{100*port_sigma_theoretical:.2f}%")

st.subheader("🧮 Pesos y parámetros por instrumento")
params_df = pd.DataFrame(
    {
        "clase": [c["clase"] for c in componentes],
        "peso": W,
        "μ_%": 100 * mu,
        "σ_%": 100 * sigmas,
    }
)
st.dataframe(params_df, use_container_width=True)

st.subheader("🧩 Matriz de Covarianza (anual)")
st.dataframe(pd.DataFrame(Cov, columns=params_df["clase"], index=params_df["clase"]).round(6), use_container_width=True)

st.subheader("💰 Ganancia generada por clase (según plazo)")
agg = (
    pd.DataFrame(
        {
            "clase": [c["clase"] for c in componentes],
            "ganancia_generada": [c["ganancia_generada"] for c in componentes],
        }
    )
    .groupby("clase", as_index=False)
    .sum()
)
st.dataframe(agg, use_container_width=True)

st.caption(
    """
    **Interpretación**
    - *Rendimiento esperado (μ)* y *σ* provienen de simulación Monte Carlo bajo supuestos paramétricos.
    - *VaR/CVaR* expresan la **pérdida porcentual** esperada al nivel de confianza seleccionado, en un horizonte anual.
    - *Beta, Alfa, R²* se calculan vs. un **benchmark** paramétrico que configuras en la barra lateral.
    - Puedes exportar las tablas desde el menú de 3 puntos arriba a la derecha de cada una.
    """
)

# =============================
# Descarga opcional de insumos/outputs como CSV
# =============================
exp1 = st.download_button(
    "⬇️ Descargar resumen instrumentos (CSV)",
    data=resumen_df.to_csv(index=False).encode("utf-8"),
    file_name="resumen_instrumentos.csv",
    mime="text/csv",
)
exp2 = st.download_button(
    "⬇️ Descargar parámetros y pesos (CSV)",
    data=params_df.to_csv(index=False).encode("utf-8"),
    file_name="parametros_pesos.csv",
    mime="text/csv",
)
exp3 = st.download_button(
    "⬇️ Descargar matriz de covarianza (CSV)",
    data=pd.DataFrame(Cov, columns=params_df["clase"], index=params_df["clase"]).to_csv().encode("utf-8"),
    file_name="covarianza.csv",
    mime="text/csv",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Cómo desplegar en Streamlit Cloud**
    1) Crea un repo en GitHub con este archivo como `app.py`.
    2) Añade `requirements.txt` con:
       
       ```
       streamlit
       pandas
       numpy
       ```
    3) En [share.streamlit.io](https://share.streamlit.io) conecta tu GitHub y elige el repo y la rama.
    4) Archivo principal: `app.py`. ¡Listo!
    """
)






