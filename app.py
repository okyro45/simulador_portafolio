# app.py
# -*- coding: utf-8 -*-

import io
import numpy as np
import pandas as pd
from dataclasses import dataclass
import streamlit as st
from typing import Dict, List, Optional
from scipy.stats import norm
import statsmodels.api as sm

# ----------------------- Utilidades Generales ----------------------- #
PERIODS_PER_YEAR = 12  # se asume periodicidad mensual


def annual_to_period_rate(r_annual: float, m: int = PERIODS_PER_YEAR) -> float:
    return (1 + r_annual) ** (1 / m) - 1


def annual_to_period_vol(vol_annual: float, m: int = PERIODS_PER_YEAR) -> float:
    return vol_annual / np.sqrt(m)


def to_annual_return(r_period: float, m: int = PERIODS_PER_YEAR) -> float:
    return (1 + r_period) ** m - 1


def to_annual_vol(vol_period: float, m: int = PERIODS_PER_YEAR) -> float:
    return vol_period * np.sqrt(m)


# Helpers robustos para convertir números que pueden venir como NaN/strings

def get_float(row_or_val, key: Optional[str] = None, default: float = 0.0) -> float:
    """Devuelve float seguro desde un DataFrame row (usando 'key') o desde un valor directo."""
    if key is not None:
        val = row_or_val.get(key, default)
    else:
        val = row_or_val
    val = pd.to_numeric(val, errors="coerce")
    return float(val) if pd.notna(val) else float(default)


def get_int(row_or_val, key: Optional[str] = None, default: int = 0) -> int:
    """Devuelve int seguro (evita ValueError: cannot convert float NaN to integer)."""
    if key is not None:
        val = row_or_val.get(key, default)
    else:
        val = row_or_val
    val = pd.to_numeric(val, errors="coerce")
    return int(val) if pd.notna(val) else int(default)


# Conversión de retornos a moneda base
# r_fx es el retorno del USD vs PEN (si r_fx > 0, el USD se apreció vs PEN)

def convert_return_to_base(
    r_local: pd.Series,
    asset_ccy: str,
    base_ccy: str,
    r_fx_usd_pen: Optional[pd.Series],
) -> pd.Series:
    r_local = r_local.astype(float)
    if asset_ccy == base_ccy:
        return r_local
    if r_fx_usd_pen is None:
        # Sin serie FX: aproximación sin convertir (advertiremos en UI)
        return r_local
    # Alinea FX al índice de r_local
    r_fx_usd_pen = pd.Series(r_fx_usd_pen).reindex(r_local.index)
    # Rellena faltantes de FX con 0 (sin movimiento) para no propagar NaN
    r_fx_usd_pen = r_fx_usd_pen.fillna(0.0)
    if asset_ccy == 'USD' and base_ccy == 'PEN':
        return (1 + r_local) * (1 + r_fx_usd_pen) - 1
    if asset_ccy == 'PEN' and base_ccy == 'USD':
        return (1 + r_local) / (1 + r_fx_usd_pen) - 1
    # Si se agregan otras monedas en el futuro
    return r_local


# VaR y CVaR (histórico y paramétrico normal)

def historical_var(returns: pd.Series, alpha: float = 0.95) -> float:
    # Pérdida positiva -> signo negativo del cuantil
    q = returns.quantile(1 - alpha)
    return -q


def historical_cvar(returns: pd.Series, alpha: float = 0.95) -> float:
    threshold = returns.quantile(1 - alpha)
    tail = returns[returns <= threshold]
    if len(tail) == 0:
        return 0.0
    return -tail.mean()


def parametric_var(mean: float, vol: float, alpha: float = 0.95) -> float:
    z = norm.ppf(1 - alpha)
    return -(mean + z * vol)


def parametric_cvar(mean: float, vol: float, alpha: float = 0.95) -> float:
    z = norm.ppf(1 - alpha)
    pdf = norm.pdf(z)
    cvar = -(mean - vol * pdf / (1 - alpha))
    return cvar


# Beta, Alfa y R^2 por regresión lineal

def regression_metrics(port_ret: pd.Series, bench_ret: pd.Series):
    df = pd.concat([port_ret, bench_ret], axis=1, join='inner').dropna()
    if df.shape[0] < 3:
        return np.nan, np.nan, np.nan
    y = df.iloc[:, 0]
    x = sm.add_constant(df.iloc[:, 1])
    model = sm.OLS(y, x).fit()
    alpha = model.params.get('const', np.nan)
    beta = model.params.iloc[1] if len(model.params) > 1 else np.nan
    r2 = model.rsquared
    return alpha, beta, r2


# ----------------------- Datos & Estructuras ----------------------- #
ASSET_CLASSES = [
    'Acciones',
    'Bonos',
    'Depósitos a Plazo',
    'Fondos Inmobiliarios',
]

CURRENCIES = ['PEN', 'USD']


@dataclass
class Position:
    symbol: str
    asset_class: str
    currency: str
    price: float = 0.0  # para acciones
    quantity: float = 0.0  # para acciones
    dividend_per_share: float = 0.0  # anual esperado, para acciones
    amount: float = 0.0  # para bonos/DPF/FII (monto invertido o valor nominal)
    rate_annual: float = 0.0  # tasa esperada anual (para bonos/DPF/FII)
    term_months: int = 12
    maturity_date: str = ''
    issuer: str = ''
    vol_annual: float = 0.15  # volatilidad anual estimada (para simulación si no hay series)


# ----------------------- UI ----------------------- #
st.set_page_config(page_title='Simulador de Portafolio PEN-USD', layout='wide')
st.title('📊 Simulador de Portafolio (PEN & USD)')

with st.sidebar:
    st.header('⚙️ Configuración')
    base_ccy = st.selectbox('Moneda base del portafolio', CURRENCIES, index=0)
    rf_annual = st.number_input('Tasa libre de riesgo anual (ej. 0.04 = 4%)', value=0.03, step=0.005, format='%.4f')
    alpha_level = st.slider('Nivel de confianza para VaR/CVaR', min_value=0.80, max_value=0.99, value=0.95, step=0.01)
    st.caption('Periodicidad asumida: mensual')

    st.subheader('🔁 Tipo de Cambio (USD vs PEN)')
    spot_fx = st.number_input('Spot USD/PEN actual (PEN por 1 USD)', value=3.80, step=0.01)
    fx_returns_file = st.file_uploader('CSV retornos USD vs PEN (columna: usd_pen)', type=['csv'], key='fxfile')

    r_fx_series = None
    if fx_returns_file is not None:
        fx_df = pd.read_csv(fx_returns_file)
        # Si existe una columna fecha, úsala como índice de la serie FX
        date_cols = [c for c in fx_df.columns if c.lower() in ('date', 'fecha')]
        if 'usd_pen' in fx_df.columns:
            r_fx_series = fx_df['usd_pen']
            if date_cols:
                try:
                    idx = pd.to_datetime(fx_df[date_cols[0]])
                    r_fx_series.index = idx
                except Exception:
                    pass
        else:
            st.warning("El CSV de FX debe tener una columna llamada 'usd_pen'. Se ignorará la serie.")

st.markdown('---')

# Plantillas de posiciones por clase de activo
st.subheader('🧾 Posiciones del Portafolio')

col1, col2 = st.columns(2)
with col1:
    st.markdown('**Acciones**')
    stocks_df = st.data_editor(
        pd.DataFrame({
            'symbol': ['ACCION1'],
            'currency': ['USD'],
            'price': [10.0],
            'quantity': [100.0],
            'dividend_per_share_annual': [0.5],
            'vol_annual': [0.25],
        }),
        num_rows='dynamic',
        key='stocks',
        use_container_width=True
    )

with col2:
    st.markdown('**Bonos**')
    bonds_df = st.data_editor(
        pd.DataFrame({
            'symbol': ['BONO1'],
            'currency': ['PEN'],
            'amount': [10000.0],
            'rate_annual': [0.08],
            'term_months': [36],
            'maturity_date': ['2030-12-31'],
            'issuer': ['EMISOR_X'],
            'vol_annual': [0.05],
        }),
        num_rows='dynamic',
        key='bonds',
        use_container_width=True
    )

col3, col4 = st.columns(2)
with col3:
    st.markdown('**Depósitos a Plazo**')
    td_df = st.data_editor(
        pd.DataFrame({
            'symbol': ['DPF1'],
            'currency': ['PEN'],
            'amount': [20000.0],
            'rate_annual': [0.06],
            'term_months': [12],
            'maturity_date': ['2026-06-30'],
            'issuer': ['BANCO_Y'],
            'vol_annual': [0.01],
        }),
        num_rows='dynamic',
        key='tds',
        use_container_width=True
    )

with col4:
    st.markdown('**Fondos de Inversión Inmobiliarios (FII)**')
    fii_df = st.data_editor(
        pd.DataFrame({
            'symbol': ['FII1'],
            'currency': ['USD'],
            'amount': [15000.0],
            'rate_annual': [0.10],
            'term_months': [60],
            'maturity_date': ['2032-01-01'],
            'issuer': ['FONDO_Z'],
            'vol_annual': [0.12],
        }),
        num_rows='dynamic',
        key='fii',
        use_container_width=True
    )

st.markdown('---')

st.subheader('📥 Series de Retornos (opcional)')
ret_file = st.file_uploader('CSV de retornos por activo (columnas = símbolos, puede incluir "benchmark")', type=['csv'], key='rets')
benchmark_provided = False
returns_df = None
if ret_file is not None:
    try:
        returns_df = pd.read_csv(ret_file, index_col=0)
        returns_df.index = pd.to_datetime(returns_df.index)
        benchmark_provided = 'benchmark' in returns_df.columns
        st.success(f'Series cargadas: {returns_df.shape[0]} períodos, {returns_df.shape[1]} columnas')
    except Exception as e:
        st.error(f'No se pudo leer el CSV de retornos: {e}')

st.markdown('---')

# Parámetros de generación sintética
with st.expander('⚗️ Si no hay series: parámetros para series sintéticas'):
    synth_periods = st.number_input('Nro. de períodos a simular (meses)', min_value=12, value=60, step=12)
    bench_mu_ann = st.number_input('Benchmark: rendimiento anual esperado', value=0.08, step=0.01, format='%.4f')
    bench_vol_ann = st.number_input('Benchmark: volatilidad anual', value=0.18, step=0.01, format='%.4f')


# ----------------------- Construcción del Portafolio ----------------------- #

def build_positions() -> pd.DataFrame:
    rows = []
    # Acciones
    if len(stocks_df) > 0:
        for _, r in stocks_df.iterrows():
            price = get_float(r, 'price', 0)
            qty = get_float(r, 'quantity', 0)
            value = float(price * qty)
            rows.append({
                'symbol': str(r.get('symbol', '')),
                'class': 'Acciones',
                'currency': str(r.get('currency', 'USD')),
                'value_nominal': value,
                'price': price,
                'quantity': qty,
                'div_ps_ann': get_float(r, 'dividend_per_share_annual', 0),
                'rate_annual': np.nan,
                'vol_annual': get_float(r, 'vol_annual', 0.20),
            })
    # Bonos
    if len(bonds_df) > 0:
        for _, r in bonds_df.iterrows():
            rows.append({
                'symbol': str(r.get('symbol', '')),
                'class': 'Bonos',
                'currency': str(r.get('currency', 'PEN')),
                'value_nominal': get_float(r, 'amount', 0),
                'rate_annual': get_float(r, 'rate_annual', 0.0),
                'term_months': get_int(r, 'term_months', 12),  # ✅ evita ValueError por NaN
                'maturity_date': str(r.get('maturity_date', '')),
                'issuer': str(r.get('issuer', '')),
                'vol_annual': get_float(r, 'vol_annual', 0.05),
            })
    # Depósitos a plazo
    if len(td_df) > 0:
        for _, r in td_df.iterrows():
            rows.append({
                'symbol': str(r.get('symbol', '')),
                'class': 'Depósitos a Plazo',
                'currency': str(r.get('currency', 'PEN')),
                'value_nominal': get_float(r, 'amount', 0),
                'rate_annual': get_float(r, 'rate_annual', 0.0),
                'term_months': get_int(r, 'term_months', 12),  # ✅ evita ValueError por NaN
                'maturity_date': str(r.get('maturity_date', '')),
                'issuer': str(r.get('issuer', '')),
                'vol_annual': get_float(r, 'vol_annual', 0.01),
            })
    # FII
    if len(fii_df) > 0:
        for _, r in fii_df.iterrows():
            rows.append({
                'symbol': str(r.get('symbol', '')),
                'class': 'Fondos Inmobiliarios',
                'currency': str(r.get('currency', 'USD')),
                'value_nominal': get_float(r, 'amount', 0),
                'rate_annual': get_float(r, 'rate_annual', 0.0),
                'term_months': get_int(r, 'term_months', 12),  # ✅ evita ValueError por NaN
                'maturity_date': str(r.get('maturity_date', '')),
                'issuer': str(r.get('issuer', '')),
                'vol_annual': get_float(r, 'vol_annual', 0.12),
            })
    df = pd.DataFrame(rows)
    return df


positions_df = build_positions()

if positions_df.empty:
    st.warning('Agrega al menos una posición.')
    st.stop()


# Valor de mercado por posición (en moneda base)
# Para acciones, usamos price*quantity convertido al spot si hay distinta moneda
# Para otros, usamos value_nominal como valor actual (aprox.)

def position_value_base(row) -> float:
    val_local = row.get('value_nominal', 0.0)
    if row['class'] == 'Acciones':
        val_local = float(get_float(row, 'price', 0.0) * get_float(row, 'quantity', 0.0))
    if row['currency'] == base_ccy:
        return float(val_local) if pd.notna(val_local) else 0.0
    # convertir al spot
    if row['currency'] == 'USD' and base_ccy == 'PEN':
        return float(val_local) * spot_fx if pd.notna(val_local) else 0.0
    if row['currency'] == 'PEN' and base_ccy == 'USD':
        return float(val_local) / spot_fx if pd.notna(val_local) and spot_fx != 0 else 0.0
    return float(val_local) if pd.notna(val_local) else 0.0


positions_df['value_base'] = positions_df.apply(position_value_base, axis=1)
positions_df['value_base'] = positions_df['value_base'].fillna(0.0)

portfolio_value = float(positions_df['value_base'].sum())

if portfolio_value > 0 and np.isfinite(portfolio_value):
    positions_df['weight'] = positions_df['value_base'] / portfolio_value
else:
    positions_df['weight'] = 0.0
    st.warning('El valor total del portafolio es 0. Verifica cantidades/precios/montos.')

st.subheader('📦 Resumen de Posiciones (en moneda base)')
st.dataframe(positions_df[['symbol', 'class', 'currency', 'value_base', 'weight']].round(4), use_container_width=True)
st.metric('Valor total del portafolio (base)', f"{portfolio_value:,.2f} {base_ccy}")

# ----------------------- Series de Retornos por Activo ----------------------- #

symbols = positions_df['symbol'].tolist()

# 1) Tomar series cargadas si existen
asset_returns_base: Dict[str, pd.Series] = {}
benchmark_series = None

if returns_df is not None:
    for sym in symbols:
        if sym in returns_df.columns:
            # Asumimos que series están en la moneda del activo; convertir a base
            r_local = returns_df[sym].astype(float)
            asset_ccy = positions_df.loc[positions_df['symbol'] == sym, 'currency'].values[0]
            r_conv = convert_return_to_base(r_local, asset_ccy, base_ccy, r_fx_series)
            asset_returns_base[sym] = r_conv
    if 'benchmark' in returns_df.columns:
        benchmark_series = returns_df['benchmark'].astype(float)

# 2) Generar series sintéticas para los activos que falten
need_synth = [s for s in symbols if s not in asset_returns_base]
if len(need_synth) > 0:
    st.info(f'Se generarán series sintéticas para: {", ".join(need_synth)}')
    # para cada activo, usar su tasa esperada y volatilidad
    for sym in need_synth:
        row = positions_df[positions_df['symbol'] == sym].iloc[0]
        if row['class'] == 'Acciones':
            # aproximación: rendimiento esperado anual por dividendo/valor
            price = float(get_float(row, 'price', 1)) or 1.0
            div = float(get_float(row, 'div_ps_ann', 0))
            mu_ann = div / price  # solo dividend yield (simplificado)
        else:
            mu_ann = float(get_float(row, 'rate_annual', 0))
        vol_ann = float(get_float(row, 'vol_annual', 0.15))
        mu_p = annual_to_period_rate(mu_ann)
        vol_p = annual_to_period_vol(vol_ann)
        np.random.seed(abs(hash(sym)) % (2**32))
        r = np.random.normal(loc=mu_p, scale=vol_p, size=int(synth_periods))
        # ✅ Cambiado 'M' (deprecado) por 'ME' (Month-End)
        idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=int(synth_periods), freq='ME')
        ser_local = pd.Series(r, index=idx, name=sym)
        # convertir a base si la moneda difiere y hay serie FX
        ccy = row['currency']
        ser_base = convert_return_to_base(ser_local, ccy, base_ccy, r_fx_series)
        asset_returns_base[sym] = ser_base

# Benchmark sintético si no hay
if benchmark_series is None:
    mu_b = annual_to_period_rate(bench_mu_ann)
    vol_b = annual_to_period_vol(bench_vol_ann)
    np.random.seed(42)
    r = np.random.normal(loc=mu_b, scale=vol_b, size=int(synth_periods))
    idx = None
    # alinear con cualquiera serie existente
    if len(asset_returns_base) > 0:
        any_ser = list(asset_returns_base.values())[0]
        idx = any_ser.index
        if len(idx) != int(synth_periods):
            # crear índice que encaje con el mínimo largo común
            n = min(len(any_ser), int(synth_periods))
            r = r[-n:]
            idx = any_ser.index[-n:]
    else:
        idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=int(synth_periods), freq='ME')  # ✅ ME
    benchmark_series = pd.Series(r, index=idx, name='benchmark')

# Alinear todo a un dataframe conjunto
all_returns = pd.DataFrame(asset_returns_base)
if all_returns.empty:
    st.error('No hay series de retornos (ni cargadas ni sintéticas). Revisa los parámetros.')
    st.stop()

# Alinear benchmark
common_index = all_returns.index
benchmark_series = benchmark_series.reindex(common_index).dropna()
if benchmark_series.empty:
    st.error('La serie de benchmark quedó vacía tras alinear fechas. Revisa tus datos/fechas.')
    st.stop()
all_returns = all_returns.loc[benchmark_series.index]

# Ponderaciones actuales
weights = (
    positions_df.set_index('symbol')['weight']
    .reindex(all_returns.columns)
    .fillna(0.0)
    .values
)

# Retornos del portafolio
port_ret = (all_returns @ weights)

# ----------------------- Métricas del Portafolio ----------------------- #
rf_p = annual_to_period_rate(rf_annual)

# rendimiento esperado y desviación estándar (period)
mu_p = port_ret.mean()
vol_p = port_ret.std(ddof=1)

mu_ann = to_annual_return(mu_p)
vol_ann = to_annual_vol(vol_p)

# Sharpe (anual)
excess_p = mu_p - rf_p
sharpe_ann = (to_annual_return(mu_p) - rf_annual) / (vol_ann + 1e-12) if vol_ann > 0 else np.nan

# Beta, Alfa, R^2
alpha_p, beta_p, r2_p = regression_metrics(port_ret, benchmark_series)

# VaR/CVaR (histórico sobre period)
var_hist = historical_var(port_ret, alpha=alpha_level)
cvar_hist = historical_cvar(port_ret, alpha=alpha_level)

# Paramétrico normal (period)
var_norm = parametric_var(mu_p, vol_p, alpha=alpha_level)
cvar_norm = parametric_cvar(mu_p, vol_p, alpha=alpha_level)

# Covarianza entre activos (period)
cov_mat = all_returns.cov()

# ----------------------- Salidas ----------------------- #
st.subheader('📈 Métricas del Portafolio')
colA, colB, colC, colD = st.columns(4)
colA.metric('Rendimiento esperado anual', f"{mu_ann:.2%}")
colB.metric('Desviación estándar anual', f"{vol_ann:.2%}")
colC.metric('Sharpe (anual)', f"{sharpe_ann:.3f}")
colD.metric('Beta vs Benchmark', f"{beta_p:.3f}")

colE, colF, colG = st.columns(3)
colE.metric('Alfa (periodo)', f"{alpha_p:.3%}")
colF.metric('R^2', f"{r2_p:.3f}")
colG.metric(f'VaR hist. {int(alpha_level*100)}% (periodo)', f"{var_hist:.2%}")

colH, colI = st.columns(2)
colH.metric(f'CVaR hist. {int(alpha_level*100)}% (periodo)', f"{cvar_hist:.2%}")
colI.metric(f'VaR Normal {int(alpha_level*100)}% (periodo)', f"{var_norm:.2%}")

st.caption('Nota: VaR/CVaR mostrados en periodicidad de las series (mensual por defecto).')

st.subheader('🧮 Matriz de Covarianza (periodo)')
st.dataframe(cov_mat.round(6), use_container_width=True)

# ----------------------- Monitoreo & Rebalanceo ----------------------- #
st.subheader('🧭 Monitoreo y Rebalanceo')
# Objetivos por clase de activo
current_by_class = positions_df.groupby('class')['value_base'].sum()
current_weights_class = current_by_class / portfolio_value if portfolio_value > 0 else current_by_class * 0

st.markdown('**Pesos actuales por clase de activo**')
st.dataframe(current_weights_class.to_frame('peso_actual').round(4), use_container_width=True)

st.markdown('**Objetivos de peso por clase de activo**')
# editor con defaults uniformes
default_targets = pd.Series(1/len(ASSET_CLASSES), index=ASSET_CLASSES)
user_targets_df = st.data_editor(
    default_targets.to_frame('peso_objetivo').reset_index().rename(columns={'index':'class'}),
    num_rows='dynamic',
    use_container_width=True,
    key='targets'
)
user_targets = user_targets_df.set_index('class')['peso_objetivo']
user_targets = user_targets.reindex(ASSET_CLASSES).fillna(0)

# normalizar
if user_targets.sum() > 0:
    user_targets = user_targets / user_targets.sum()

rebalance_tol = st.slider('Tolerancia de rebalanceo por clase (±%)', 0.0, 0.20, 0.05, 0.01)

# Calcular diferencias y propuestas
rebalance_df = pd.DataFrame({
    'peso_actual': current_weights_class.reindex(ASSET_CLASSES).fillna(0),
    'peso_objetivo': user_targets,
})
rebalance_df['desviacion'] = rebalance_df['peso_actual'] - rebalance_df['peso_objetivo']
rebalance_df['dentro_rango'] = rebalance_df['desviacion'].abs() <= rebalance_tol

# Monto a comprar/vender por clase
rebalance_df['monto_requerido'] = (rebalance_df['peso_objetivo'] - rebalance_df['peso_actual']) * portfolio_value

st.dataframe(rebalance_df.round(4), use_container_width=True)

# Propuesta proporcional por símbolo:
# distribuir el monto a comprar/vender dentro de cada clase según su peso relativo
proposals = []
for cls in ASSET_CLASSES:
    delta_cls = rebalance_df.loc[cls, 'monto_requerido'] if cls in rebalance_df.index else 0.0
    if abs(delta_cls) < 1e-8:
        continue
    cls_symbols = positions_df[positions_df['class'] == cls]
    if cls_symbols.empty:
        continue
    # pesos relativos dentro de la clase (maneja suma 0)
    sum_cls = cls_symbols['value_base'].sum()
    if sum_cls > 0:
        w_rel = cls_symbols['value_base'] / sum_cls
    else:
        w_rel = pd.Series(1.0 / len(cls_symbols), index=cls_symbols.index)
    for idx_row, r in cls_symbols.iterrows():
        amt = delta_cls * w_rel.loc[idx_row]
        proposals.append({
            'class': cls,
            'symbol': r['symbol'],
            'currency': r['currency'],
            'monto_en_base': amt,
        })

proposals_df = pd.DataFrame(proposals)
if not proposals_df.empty:
    st.markdown('**Propuesta de rebalanceo por símbolo (monto en moneda base)**')
    st.dataframe(proposals_df, use_container_width=True)
else:
    st.info('No hay diferencias significativas vs objetivos dentro de la tolerancia.')

# ----------------------- Descargas ----------------------- #
st.subheader('⬇️ Exportar')
# Reporte en Excel con hojas: posiciones, métricas, covarianza, rebalanceo
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine='openpyxl') as writer:
    positions_df.to_excel(writer, sheet_name='posiciones', index=False)
    metrics_df = pd.DataFrame({
        'metrica': [
            'mu_anual', 'vol_anual', 'sharpe_anual', 'alpha_periodo', 'beta', 'r2',
            f'var_hist_{int(alpha_level*100)}%', f'cvar_hist_{int(alpha_level*100)}%',
            f'var_norm_{int(alpha_level*100)}%', f'cvar_norm_{int(alpha_level*100)}%'
        ],
        'valor': [mu_ann, vol_ann, sharpe_ann, alpha_p, beta_p, r2_p, var_hist, cvar_hist, var_norm, cvar_norm]
    })
    metrics_df.to_excel(writer, sheet_name='metricas', index=False)
    cov_mat.to_excel(writer, sheet_name='covarianza')
    rebalance_df.to_excel(writer, sheet_name='rebalanceo')
    if not proposals_df.empty:
        proposals_df.to_excel(writer, sheet_name='propuestas', index=False)

excel_bytes = buf.getvalue()
st.download_button('Descargar reporte Excel', data=excel_bytes, file_name='reporte_portafolio.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

st.caption('Sugerencias: carga tus series históricas para métricas más realistas. Asegúrate de incluir una columna "benchmark" y, si trabajas en moneda cruzada, la serie de retornos de USD vs PEN (columna "usd_pen").')





