"""
FRED Retail Sales Forecasting — Streamlit Dashboard
Deploy: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import calendar
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="FRED Retail Sales Forecaster",
    page_icon="📈",
    layout="wide",
)

# ── Colour palette ─────────────────────────────────────────────────────────────
C = dict(train="#2563EB", test="#16A34A", forecast="#DC2626",
         future="#9333EA", ci="rgba(147,51,234,0.15)")


# ── Data ───────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching RSXFS from FRED …")
def load_data():
    try:
        import pandas_datareader.data as web
        from datetime import datetime
        df = web.DataReader("RSXFS", "fred", datetime(1992, 1, 1), datetime(2025, 1, 1))
        df.columns = ["retail_sales"]
        return df, "FRED API"
    except Exception:
        # Synthetic fallback so the app still runs without an API key
        dates = pd.date_range("1992-01-01", "2025-01-01", freq="MS")
        n = len(dates)
        trend    = np.linspace(140_000, 680_000, n)
        seasonal = 22_000 * np.sin(2 * np.pi * np.arange(n) / 12 - np.pi / 2)
        noise    = np.random.default_rng(42).normal(0, 6_000, n)
        df = pd.DataFrame({"retail_sales": trend + seasonal + noise}, index=dates)
        return df, "Synthetic (FRED API key not set)"


df, source = load_data()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")
    st.caption(f"Data source: **{source}**")
    st.markdown("---")

    page = st.radio("Page", ["📊 Overview", "🔀 Decomposition",
                              "📐 Stationarity & ACF/PACF",
                              "🔮 SARIMA Forecast"])
    st.markdown("---")

    if "Forecast" in page:
        holdout = st.slider("Hold-out months (test set)", 12, 36, 24, step=6)
        horizon = st.slider("Forecast horizon (months)", 6, 24, 12, step=6)
        sarima_p = st.selectbox("AR order p", [0, 1, 2], index=1)
        sarima_q = st.selectbox("MA order q", [0, 1, 2], index=1)
        ci_level = st.select_slider("Confidence interval", ["80%", "90%", "95%"], value="90%")
        alpha    = 1 - int(ci_level[:-1]) / 100

    st.markdown("---")
    st.markdown("📚 [FRED RSXFS](https://fred.stlouisfed.org/series/RSXFS)")
    st.markdown("👤 [Vikash Maheshwari](https://vikash-maheshwari.vercel.app/)")


# ── Page: Overview ─────────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.title("📈 U.S. Monthly Retail Sales (RSXFS)")
    st.caption("Advance Retail Sales excl. Food Services — Federal Reserve Bank of St. Louis")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Observations",  f"{len(df):,}")
    col2.metric("Latest (Jan 2025)", f"${df['retail_sales'].iloc[-1]:,.0f}M")
    col3.metric("Jan 1992 Baseline",  f"${df['retail_sales'].iloc[0]:,.0f}M")
    growth = (df['retail_sales'].iloc[-1] / df['retail_sales'].iloc[0] - 1) * 100
    col4.metric("Total Growth", f"{growth:.0f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df['retail_sales'],
        mode='lines', name='Retail Sales',
        line=dict(color=C['train'], width=2),
        fill='tozeroy', fillcolor='rgba(37,99,235,0.08)'
    ))
    # Annotations
    events = {
        "2008-09": ("Financial Crisis", -40_000),
        "2020-04": ("COVID-19 Drop",    -60_000),
        "2020-06": ("V-shape Recovery",  50_000),
    }
    for d, (label, dy) in events.items():
        ts = pd.Timestamp(d)
        idx = df.index.get_indexer([ts], method='nearest')[0]
        y   = df['retail_sales'].iloc[idx]
        fig.add_annotation(x=ts, y=y + dy, text=label, showarrow=True,
                           arrowhead=2, font=dict(size=10, color="#6B7280"))

    fig.update_layout(height=420, xaxis_title="Date", yaxis_title="$ Millions",
                      margin=dict(l=0, r=0, t=20, b=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Summary Statistics"):
        st.dataframe(df['retail_sales'].describe().rename("Value").apply(lambda x: f"${x:,.0f}M"))


# ── Page: Decomposition ────────────────────────────────────────────────────────
elif page == "🔀 Decomposition":
    st.title("🔀 Seasonal Decomposition")
    st.markdown("Additive decomposition: **Original = Trend + Seasonal + Residual** (period = 12)")

    decomp = seasonal_decompose(df['retail_sales'], model='additive', period=12)
    seasonal_avg = decomp.seasonal.groupby(decomp.seasonal.index.month).mean()

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=["Original", "Trend", "Seasonal", "Residual"])
    pairs = [
        (df['retail_sales'], C['train'], 1),
        (decomp.trend,       C['forecast'], 2),
        (decomp.seasonal,    C['test'], 3),
        (decomp.resid,       C['future'], 4),
    ]
    for data, color, row in pairs:
        fig.add_trace(go.Scatter(x=data.index, y=data, mode='lines',
                                 line=dict(color=color, width=1.5), showlegend=False), row=row, col=1)
    fig.update_layout(height=750, margin=dict(l=0, r=0, t=60, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📅 Average Seasonal Effect by Month")
    months = [calendar.month_abbr[m] for m in seasonal_avg.index]
    fig2 = go.Figure(go.Bar(x=months, y=seasonal_avg.values,
                             marker_color=[C['forecast'] if v < 0 else C['test'] for v in seasonal_avg.values]))
    fig2.update_layout(height=300, yaxis_title="Additive Effect ($ Millions)",
                       margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)

    peak   = seasonal_avg.idxmax()
    trough = seasonal_avg.idxmin()
    c1, c2 = st.columns(2)
    c1.metric("Peak Month",   calendar.month_name[peak],   f"+${seasonal_avg[peak]:,.0f}M vs avg")
    c2.metric("Trough Month", calendar.month_name[trough], f"${seasonal_avg[trough]:,.0f}M vs avg")


# ── Page: Stationarity & ACF/PACF ─────────────────────────────────────────────
elif page == "📐 Stationarity & ACF/PACF":
    st.title("📐 Stationarity Testing & Autocorrelation")

    df['diff1'] = df['retail_sales'].diff()

    def adf(series, label):
        r = adfuller(series.dropna(), autolag='AIC')
        return {
            "Series": label, "ADF Statistic": f"{r[0]:.4f}",
            "p-value": f"{r[1]:.4f}", "Result": "✅ Stationary" if r[1] < 0.05 else "❌ Non-stationary"
        }

    results = [adf(df['retail_sales'], "Original"), adf(df['diff1'], "1st Difference")]
    st.dataframe(pd.DataFrame(results).set_index("Series"), use_container_width=True)

    st.markdown("---")
    st.subheader("ACF & PACF (1st Differenced Series)")

    from statsmodels.tsa.stattools import acf, pacf
    diff_clean = df['diff1'].dropna()
    LAGS = 36
    acf_vals,  acf_ci  = acf(diff_clean,  nlags=LAGS, alpha=0.05, fft=True)
    pacf_vals, pacf_ci = pacf(diff_clean, nlags=LAGS, alpha=0.05, method='ywm')
    lag_x = np.arange(1, LAGS + 1)

    def corr_chart(vals, ci, title):
        ci_upper = ci[1:, 1] - vals[1:]
        ci_lower = vals[1:] - ci[1:, 0]
        fig = go.Figure()
        for i, v in enumerate(vals[1:]):
            fig.add_shape(type="line", x0=lag_x[i], x1=lag_x[i], y0=0, y1=v,
                          line=dict(color=C['train'], width=2))
        fig.add_trace(go.Scatter(x=lag_x, y=vals[1:], mode='markers',
                                 marker=dict(color=C['train'], size=6), name='Corr'))
        fig.add_trace(go.Scatter(x=lag_x, y=ci[1:, 1] - vals[1:], mode='lines',
                                 line=dict(dash='dash', color='gray', width=1), name='95% CI', showlegend=False))
        fig.add_trace(go.Scatter(x=lag_x, y=ci[1:, 0] - vals[1:], mode='lines',
                                 line=dict(dash='dash', color='gray', width=1), showlegend=False,
                                 fill='tonexty', fillcolor='rgba(107,114,128,0.1)'))
        fig.add_hline(y=0, line_width=1, line_color="black")
        fig.update_layout(title=title, height=320, xaxis_title="Lag",
                          yaxis_title="Correlation", margin=dict(l=0, r=0, t=40, b=0))
        return fig

    c1, c2 = st.columns(2)
    c1.plotly_chart(corr_chart(acf_vals, acf_ci, "ACF — 1st Diff"), use_container_width=True)
    c2.plotly_chart(corr_chart(pacf_vals, pacf_ci, "PACF — 1st Diff"), use_container_width=True)

    st.info("**Reading:** PACF cuts off after lag 1 → AR(1). ACF cuts off after lag 1 → MA(1). "
            "Spikes at lags 12, 24 → seasonal AR/MA. → **SARIMA(1,1,1)(1,1,1)[12]**")


# ── Page: SARIMA Forecast ──────────────────────────────────────────────────────
elif page == "🔮 SARIMA Forecast":
    st.title("🔮 SARIMA Forecasting")
    st.markdown(f"Model: **SARIMA({sarima_p},1,{sarima_q})(1,1,1)[12]** · "
                f"Hold-out: {holdout} months · Horizon: {horizon} months · CI: {ci_level}")

    series = df['retail_sales']
    train  = series.iloc[:-holdout]
    test   = series.iloc[-holdout:]

    with st.spinner("Fitting SARIMA …"):
        model = SARIMAX(train, order=(sarima_p, 1, sarima_q),
                        seasonal_order=(1, 1, 1, 12),
                        enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)

    pred       = res.get_forecast(steps=holdout + horizon)
    pred_mean  = pred.predicted_mean
    pred_ci    = pred.conf_int(alpha=alpha)
    test_pred  = pred_mean.iloc[:holdout]
    fut_mean   = pred_mean.iloc[holdout:]
    fut_ci     = pred_ci.iloc[holdout:]

    # ── KPIs ──────────────────────────────────────────────────────────────────
    mae  = mean_absolute_error(test, test_pred)
    rmse = np.sqrt(mean_squared_error(test, test_pred))
    mape = np.mean(np.abs((test.values - test_pred.values) / test.values)) * 100
    acc  = 100 - mape

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE",              f"${mae:,.0f}M")
    m2.metric("RMSE",             f"${rmse:,.0f}M")
    m3.metric("MAPE",             f"{mape:.2f}%")
    m4.metric("Forecast Accuracy",f"{acc:.1f}%")
    st.markdown("---")

    # ── Chart ─────────────────────────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.iloc[-60:].index, y=train.iloc[-60:],
                             name='Train', line=dict(color=C['train'], width=2)))
    fig.add_trace(go.Scatter(x=test.index, y=test,
                             name='Actual (test)', line=dict(color=C['test'], width=2)))
    fig.add_trace(go.Scatter(x=test_pred.index, y=test_pred,
                             name='SARIMA (test)', line=dict(color=C['forecast'], width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=fut_mean.index, y=fut_mean,
                             name=f'{horizon}-month forecast',
                             line=dict(color=C['future'], width=2.5, dash='dash')))
    fig.add_trace(go.Scatter(
        x=pd.concat([fut_mean, fut_mean.iloc[::-1]]).index,
        y=pd.concat([fut_ci.iloc[:,1], fut_ci.iloc[:,0].iloc[::-1]]).values,
        fill='toself', fillcolor=C['ci'], line=dict(color='rgba(0,0,0,0)'),
        name=f'{ci_level} CI', showlegend=True))

    fig.add_vline(x=str(test.index[0]), line_dash='dot', line_color='gray',
                  annotation_text='Train/Test', annotation_position='top left')
    fig.add_vline(x=str(fut_mean.index[0]), line_dash='dashdot', line_color='gray',
                  annotation_text='Test/Future', annotation_position='top left')

    fig.update_layout(height=500, xaxis_title="Date", yaxis_title="$ Millions",
                      legend=dict(orientation='h', y=-0.15),
                      margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Future Forecast Table")
        fut_df = pd.DataFrame({
            "Month": fut_mean.index.strftime('%b %Y'),
            "Forecast ($M)": fut_mean.values.round(0).astype(int),
            f"Lower {ci_level}": fut_ci.iloc[:,0].values.round(0).astype(int),
            f"Upper {ci_level}": fut_ci.iloc[:,1].values.round(0).astype(int),
        })
        st.dataframe(fut_df.set_index("Month"), use_container_width=True)
    with col2:
        st.subheader("Model Info")
        st.code(f"""SARIMA({sarima_p},1,{sarima_q})(1,1,1)[12]
AIC : {res.aic:.1f}
BIC : {res.bic:.1f}
Log-likelihood: {res.llf:.1f}
Observations  : {len(train)}
Hold-out      : {holdout} months
Forecast acc. : {acc:.1f}%""")
