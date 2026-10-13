# ============================================================
# STREAMLIT — BTC FORECAST (CNN-BiLSTM + VMD, MULTI-HORIZON)
# Live data: hourly BTC/USDT candles downloaded from Binance
# at run time — no file upload needed.
# ============================================================

import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import torch
import torch.nn as nn
from vmdpy import VMD

# ============================================================
# CONFIG (overridable with environment variables)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))
MODEL_PATH = MODEL_DIR / os.getenv("MODEL_FILE", "1h_2h_4h_6h_8h_12h_24h.pth")
SCALER_PATH = MODEL_DIR / os.getenv("SCALER_FILE", "1h_2h_4h_6h_8h_12h_24h.pkl")

# Optional local CSV (1-minute candles) used only if Binance is unreachable
FALLBACK_CSV = Path(os.getenv("FALLBACK_CSV", BASE_DIR / "data" / "Val_data.csv"))

EVAL_PATH = MODEL_DIR / "evaluation.json"

# Start of the price history downloaded and shown on the chart
HISTORY_START = os.getenv("HISTORY_START", "2025-11-11")

# Causal VMD: each forecast decomposes ONLY the VMD_WINDOW hourly log-returns
# ending at the forecast time (must match train.py / features.py)
VMD_WINDOW = 256
VMD_PARAMS = dict(alpha=2000, tau=0, K=3, DC=0, init=1, tol=1e-7)
REFRESH_SECONDS = int(os.getenv("REFRESH_SECONDS", "300"))

# Public Binance market-data endpoints (no API key). The first one is
# Binance's market-data-only domain; the second is the main API.
BINANCE_ENDPOINTS = [
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
]
SYMBOL = "BTCUSDT"

SEQ_LEN = 24
HORIZONS = [1, 2, 4, 6, 8, 12, 24]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(layout="wide", page_title="BTC Forecast — CNN-BiLSTM + VMD")

# ============================================================
# MODEL
# ============================================================


class CNN_BiLSTM(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 32, 3)
        self.conv2 = nn.Conv1d(32, 64, 3)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            64, 64,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        self.fc = nn.Linear(128, out_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


@st.cache_resource
def load_model_and_scalers():
    model = CNN_BiLSTM(len(HORIZONS)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    scalers = joblib.load(SCALER_PATH)
    return model, scalers["X"], scalers["y"]

# ============================================================
# DATA
# ============================================================


def _fetch_binance_hourly(start_ms: int) -> pd.DataFrame:
    """Download every 1h candle from start_ms to now (1,000 candles per call)."""
    last_error = None
    for url in BINANCE_ENDPOINTS:
        try:
            rows, cursor = [], start_ms
            while True:
                resp = requests.get(
                    url,
                    params={"symbol": SYMBOL, "interval": "1h",
                            "startTime": cursor, "limit": 1000},
                    timeout=20,
                )
                resp.raise_for_status()
                batch = resp.json()
                if not batch:
                    break
                rows.extend(batch)
                if len(batch) < 1000:
                    break
                cursor = batch[-1][0] + 3_600_000
                time.sleep(0.1)
            if rows:
                df = pd.DataFrame(rows).iloc[:, [0, 4, 6]]
                df.columns = ["open_time", "close", "close_time"]
                return df
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
    raise RuntimeError(f"Binance unreachable: {last_error}")


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner="Downloading latest BTC prices…")
def load_hourly_closes() -> tuple[pd.DataFrame, str]:
    """
    Hourly closes, identical to resampling 1-minute closes with
    resample("1h").last(). Only completed hours are kept.
    """
    start_ms = int(pd.Timestamp(HISTORY_START, tz="UTC").timestamp() * 1000)
    try:
        raw = _fetch_binance_hourly(start_ms)
        now_ms = int(time.time() * 1000)
        raw = raw[raw["close_time"] < now_ms]          # drop the hour still in progress
        idx = pd.to_datetime(raw["open_time"], unit="ms")
        df_h = pd.DataFrame({"close": raw["close"].astype(float).values}, index=idx)
        source = "Binance BTC/USDT (live)"
    except RuntimeError:
        if not FALLBACK_CSV.exists():
            raise
        df = pd.read_csv(FALLBACK_CSV)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.set_index("datetime").sort_index()
        df_h = df["close"].resample("1h").last().ffill().to_frame("close")
        source = f"local file {FALLBACK_CSV.name} (Binance unreachable)"
    df_h.index.name = "datetime"
    df_h = df_h[~df_h.index.duplicated(keep="last")].sort_index()
    df_h = df_h.asfreq("1h").ffill()
    return df_h, source


def hourly_from_upload(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.set_index("datetime").sort_index()
    return df["close"].resample("1h").last().ffill().to_frame("close")


def add_log_returns(df_h: pd.DataFrame) -> pd.DataFrame:
    out = df_h.copy()
    out["log_return"] = np.log(out["close"] / out["close"].shift(1))
    return out.dropna()


def causal_vmd_features(log_returns: np.ndarray) -> np.ndarray:
    """VMD of the last VMD_WINDOW returns only (no future data); last SEQ_LEN rows of the 3 modes."""
    p = VMD_PARAMS
    u, _, _ = VMD(log_returns[-VMD_WINDOW:].astype(np.float64),
                  p["alpha"], p["tau"], p["K"], p["DC"], p["init"], p["tol"])
    return u.T[-SEQ_LEN:].astype(np.float32)


@st.cache_data
def load_evaluation():
    try:
        import json
        return json.loads(EVAL_PATH.read_text())
    except (OSError, ValueError):
        return None

# ============================================================
# SIDEBAR — DATA SOURCE
# ============================================================

st.sidebar.title("⏱️ Settings")

with st.sidebar.expander("Data source", expanded=False):
    uploaded = st.file_uploader(
        "Optional: use your own 1-minute CSV instead "
        "(columns: timestamp, open, high, low, close, volume)",
        type=["csv", "txt"],
    )
    if st.button("Refresh prices now"):
        load_hourly_closes.clear()

try:
    if uploaded is not None:
        df_hourly, data_source = hourly_from_upload(uploaded), f"uploaded file {uploaded.name}"
    else:
        df_hourly, data_source = load_hourly_closes()
except Exception as exc:  # noqa: BLE001
    st.error(
        "Price data could not be loaded from Binance. "
        "Upload a 1-minute CSV in the sidebar (Data source) to continue."
    )
    st.caption(f"Details: {exc}")
    st.stop()

df_h = add_log_returns(df_hourly)
if len(df_h) <= VMD_WINDOW + 1:
    st.error(f"At least {VMD_WINDOW + 2} hours of prices are needed.")
    st.stop()

try:
    model, scaler_X, scaler_y = load_model_and_scalers()
except FileNotFoundError as exc:
    st.error(f"Model files not found in {MODEL_DIR}. Details: {exc}")
    st.stop()

# ============================================================
# SIDEBAR — WINDOW & HORIZON
# ============================================================

start_dt, end_dt = st.sidebar.slider(
    "Observation window",
    min_value=df_h.index[VMD_WINDOW].to_pydatetime(),
    max_value=df_h.index[-1].to_pydatetime(),
    value=(df_h.index[max(VMD_WINDOW, len(df_h) - 200)].to_pydatetime(),
           df_h.index[-1].to_pydatetime()),
    step=pd.Timedelta(hours=1),
    format="YYYY-MM-DD HH:mm"
)

horizon_sel = st.sidebar.selectbox(
    "Maximum forecast horizon (h)",
    HORIZONS,
    index=HORIZONS.index(4)
)

st.sidebar.caption(
    f"Source: {data_source}  \n"
    f"Last completed hour: {df_h.index[-1]:%Y-%m-%d %H:%M} UTC  \n"
    f"Prices refresh every {REFRESH_SECONDS // 60} min."
)

# ============================================================
# INFERENCE
# ============================================================

df_window = df_h.loc[start_dt:end_dt]
last_price = df_window["close"].iloc[-1]

# Only returns up to the end of the observation window are used (no look-ahead)
past_returns = df_h.loc[:end_dt, "log_return"].values
X = scaler_X.transform(causal_vmd_features(past_returns))
X = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    lr_preds = scaler_y.inverse_transform(model(X).cpu().numpy())[0]

# ============================================================
# BUILD FORECAST (WITH CONNECTION)
# ============================================================

dates = [df_window.index[-1]]
prices = [last_price]
labels = ["Now"]

p = last_price

for h in HORIZONS:
    if h > horizon_sel:
        break
    p = p * np.exp(lr_preds[HORIZONS.index(h)])
    dates.append(df_window.index[-1] + pd.Timedelta(hours=h))
    prices.append(p)
    labels.append(f"H+{h}h")

# ============================================================
# PLOT
# ============================================================

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_h.index,
    y=df_h["close"],
    mode="lines",
    line=dict(color="#4fa3ff"),
    name="Observed (full)"
))

fig.add_trace(go.Scatter(
    x=df_window.index,
    y=df_window["close"],
    mode="lines+markers",
    line=dict(color="#00ffff", width=2),
    marker=dict(size=5),
    name="Observed (window)"
))

fig.add_trace(go.Scatter(
    x=dates,
    y=prices,
    mode="lines+markers+text",
    line=dict(color="orange", width=3),
    marker=dict(size=9),
    text=labels,
    textposition="top center",
    name="Forecast (CNN-BiLSTM)"
))

period = f"{df_h.index[0]:%B %Y} → {df_h.index[-1]:%d %B %Y}"
fig.update_layout(
    title=dict(
        text=(
            "<b>CNN-BiLSTM WITH CAUSAL VMD — MULTI-HORIZON BTC FORECAST<br>"
            f"BTC/USDT HOURLY CLOSES ({period})</b>"
        ),
        x=0.5,
        xanchor="center",
        font=dict(size=26)
    ),
    legend=dict(
        font=dict(size=24)
    ),
    hovermode="x unified",
    dragmode="zoom",
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font=dict(color="white"),
    xaxis=dict(showgrid=True, gridcolor="#333"),
    yaxis=dict(showgrid=True, gridcolor="#333"),
    height=850
)

st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})

# ============================================================
# OUT-OF-SAMPLE PERFORMANCE (from train.py)
# ============================================================

evaluation = load_evaluation()
if evaluation:
    with st.expander("Out-of-sample performance of the model", expanded=False):
        st.markdown(
            f"Test period never used for training or model selection: "
            f"**{evaluation['test_period']}** ({evaluation['n_test']:,} hourly forecasts). "
            "The benchmark is a random walk (forecast of zero return)."
        )
        rows = [{
            "Horizon": h,
            "RMSE model (%)": round(m["rmse_model_pct"], 3),
            "RMSE random walk (%)": round(m["rmse_random_walk_pct"], 3),
            "R² vs random walk": round(m["r2_oos_vs_random_walk"], 4),
            "Direction correct": f"{m['directional_accuracy']:.1%}",
            "Share of up moves": f"{m['share_of_up_moves']:.1%}",
        } for h, m in evaluation["horizons"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            "A negative R² means the model does not beat the random walk. Hourly Bitcoin "
            "returns are close to unpredictable; this dashboard is a research demonstration, "
            "not investment advice."
        )
