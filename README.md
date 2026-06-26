# 📈 U.S. Retail Sales — Time Series Analysis & Forecasting

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

End-to-end time series analysis and forecasting of U.S. monthly retail sales using FRED economic data. Covers classical decomposition, stationarity testing, and a SARIMA model with a 12-month ahead forecast — deployed as an interactive Streamlit dashboard.

---

## Dataset

| Field | Value |
|-------|-------|
| Series | RSXFS — Advance Retail Sales excl. Food Services |
| Source | [Federal Reserve Bank of St. Louis (FRED)](https://fred.stlouisfed.org/series/RSXFS) |
| Frequency | Monthly |
| Range | Jan 1992 – Jan 2025 |
| Observations | 403 |

---

## Analysis Pipeline

**1. EDA** — Summary statistics, null checks, distribution  
**2. Visualization** — Full time series with key event annotations (Financial Crisis, COVID-19)  
**3. Seasonal Decomposition** — Additive decomposition (trend · seasonal · residual)  
**4. Stationarity Testing** — ADF test on raw + 1st-differenced series  
**5. ACF & PACF** — Lag analysis to determine SARIMA order  
**6. SARIMA Forecasting** — SARIMA(1,1,1)(1,1,1)[12] with 24-month hold-out evaluation  
**7. Future Forecast** — 12-month ahead prediction with 90% confidence intervals  

---

## Key Findings

- **Trend:** Consistent upward growth; retail sales grew ~5× from 1992 to 2025
- **Seasonality:** December peak (holiday); January/February trough — recurring every year
- **COVID-19:** −17% drop in April 2020, full V-shape recovery by June 2020
- **Stationarity:** Non-stationary in levels; stationary after 1st differencing (ADF p < 0.001)
- **Model:** SARIMA(1,1,1)(1,1,1)[12] → MAPE ≈ 4–5% on 24-month hold-out

---

## Tech Stack

`Python` · `Pandas` · `NumPy` · `Statsmodels` · `Scikit-learn` · `Plotly` · `Streamlit` · `pandas-datareader`

---

## Run Locally

```bash
git clone https://github.com/VikashMaheshwari/FRED-Economic-Analysis.git
cd FRED-Economic-Analysis
pip install -r requirements.txt
streamlit run app.py
```

The notebook pulls data live from the FRED API via `pandas_datareader` — no manual download needed. Get a free API key at https://fred.stlouisfed.org/docs/api/api_key.html (optional; the app falls back to synthetic data if no key is set).

---

## Project Structure

```
FRED-Economic-Analysis/
├── FRED_Economic_Analysis.ipynb   # Full analysis notebook
├── app.py                         # Streamlit dashboard
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Author

**Vikash Maheshwari** · M.Eng CS&E  
[Portfolio](https://vikash-maheshwari.vercel.app/) · [LinkedIn](https://linkedin.com/in/vikashmaheshwari) · [GitHub](https://github.com/VikashMaheshwari)
