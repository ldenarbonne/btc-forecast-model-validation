# BTC Forecast Dashboard — CNN-BiLSTM + VMD

Streamlit dashboard forecasting the Bitcoin price (BTC/USDT) over 1h to 24h horizons.

- **Model**: CNN + bidirectional LSTM, multi-horizon (1h, 2h, 4h, 6h, 8h, 12h, 24h), predicting log-returns
- **Features**: causal Variational Mode Decomposition (VMD, K = 3) of the last 256 hourly log-returns, 24-hour input window
- **Training**: Binance hourly data 2018 → 2024, validation 2024-25, out-of-sample test from July 2025 (see `training/`)
- **Data**: hourly BTC/USDT closes downloaded automatically from Binance's public market-data API
  (no API key), refreshed every 5 minutes. No file upload needed.

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
