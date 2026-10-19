# BTC Forecast & Risk Dashboard — model validation study

Streamlit dashboard that forecasts Bitcoin (BTC/USDT) and, above all, **validates** the forecasts.
Live app: https://btc-model-validation.streamlit.app/ · Technical note: https://ldenarbonne.github.io/

Main findings (out-of-sample test, July 2025 → October 2026):
- **Direction is not predictable**: CNN-BiLSTM + VMD, Temporal Fusion Transformer, ARIMA/SARIMA do not beat a
  random walk in a usable way (R², MAE, RMSE, Gini/ROC, Diebold-Mariano, Pesaran-Timmermann, net Sharpe).
- **Look-ahead bias**: a VMD computed on the whole series creates a spurious 76-80% directional accuracy.
- **Risk is predictable**: a fat-tailed EWMA risk cone passes the Kupiec test, and a log-HAR model (Corsi, 2009)
  forecasts next-day realised volatility significantly better than RiskMetrics EWMA.

- **Model**: CNN + bidirectional LSTM, multi-horizon (1h, 2h, 4h, 6h, 8h, 12h, 24h), predicting log-returns
- **Features**: causal Variational Mode Decomposition (VMD, K = 3) of the last 256 hourly log-returns, 24-hour input window
- **Training**: Binance hourly data 2018 → 2024, validation 2024-25, out-of-sample test from July 2025 (see `training/`)
- **Risk cone**: EWMA volatility (λ = 0.94) with empirical fat-tailed multipliers, backtested with the Kupiec test
- **Volatility forecast**: log-HAR model on daily realised variance from 5-minute returns (1-day and 7-day horizons)
- **Data**: BTC/USDT prices downloaded automatically from Binance's public market-data API
  (no API key), hourly and 5-minute. No file upload needed.

## Project structure

```
app.py               Streamlit application
models/              Trained model (.pth), scalers (.pkl) and out-of-sample evaluation (evaluation.json)
training/            Scripts to download the data and retrain the model
data/                Optional fallback CSV (1-minute candles), used only if Binance is unreachable
requirements.txt     Python dependencies (CPU-only PyTorch on Linux)
Dockerfile           Container image for Cloud Run
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy for free on Streamlit Community Cloud

On https://share.streamlit.io: Create app → repository `ldenarbonne/btc-forecast-model-validation`,
branch `main`, main file `app.py`, Advanced settings → Python **3.12**.

## Deploy to Google Cloud Run (alternative)

```bash
gcloud run deploy btc-dashboard --source . --region europe-west1 --memory 2Gi --allow-unauthenticated --project YOUR_PROJECT_ID
```

## Configuration (optional environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `HISTORY_START` | `2025-11-11` | First date of the history used for the VMD decomposition |
| `REFRESH_SECONDS` | `300` | How often prices are re-downloaded |
| `MODEL_DIR` | `models/` | Folder containing the model and scaler files |

Author: Laurent Motais de Narbonne — https://ldenarbonne.github.io/
