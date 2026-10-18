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
COMPARISON_PATH = MODEL_DIR / "comparison_tft_vs_cnn.json"
RISK_CONE_PATH = MODEL_DIR / "risk_cone.json"

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


def _read_json(path):
    try:
        import json
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


@st.cache_data
def load_evaluation():
    return _read_json(EVAL_PATH)


@st.cache_data
def load_comparison():
    return _read_json(COMPARISON_PATH)


@st.cache_data
def load_risk_cone():
    return _read_json(RISK_CONE_PATH)

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
    value=(df_h.index[max(VMD_WINDOW, len(df_h) - 72)].to_pydatetime(),
           df_h.index[-1].to_pydatetime()),
    step=pd.Timedelta(hours=1),
    format="YYYY-MM-DD HH:mm"
)

horizon_sel = st.sidebar.selectbox(
    "Maximum forecast horizon (h)",
    HORIZONS,
    index=HORIZONS.index(24)
)

risk_cone = load_risk_cone()
show_cone = st.sidebar.checkbox("Show risk cone (68% / 95% / 99%)", value=risk_cone is not None,
                                disabled=risk_cone is None)
n_scenarios = st.sidebar.slider("Simulated scenarios", 0, 20, 5, disabled=risk_cone is None)

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
# RISK CONE, SCENARIOS AND REALISED PATH
# ============================================================
# EWMA volatility (RiskMetrics, lambda = 0.94) known at the end of the window
EWMA_LAMBDA = risk_cone["lambda"] if risk_cone else 0.94
past_r = df_h.loc[:end_dt, "log_return"]
sigma_now = float(np.sqrt(past_r.pow(2).ewm(alpha=1 - EWMA_LAMBDA, adjust=False).mean().iloc[-1]))
t0 = df_window.index[-1]
cone_hours = np.arange(0, horizon_sel + 1)
cone_times = [t0 + pd.Timedelta(hours=int(h)) for h in cone_hours]
cone = {}
if risk_cone and show_cone:
    for q in ("0.99", "0.95", "0.68"):
        c = np.array([0.0] + [risk_cone["multipliers"][str(h)][q] for h in cone_hours[1:]])
        width = c * sigma_now * np.sqrt(cone_hours)
        cone[q] = (last_price * np.exp(-width), last_price * np.exp(width))

scenarios = []
if risk_cone and n_scenarios > 0:
    rng = np.random.default_rng(int(pd.Timestamp(t0).timestamp()) // 3600)
    z = np.array(risk_cone["standardised_hourly_returns"])
    draws = rng.choice(z, size=(n_scenarios, horizon_sel)) * sigma_now
    scenarios = last_price * np.exp(np.concatenate([np.zeros((n_scenarios, 1)), np.cumsum(draws, axis=1)], axis=1))

realised = df_h.loc[t0:t0 + pd.Timedelta(hours=horizon_sel), "close"]
realised = realised if len(realised) > 1 else None

# ============================================================
# PLOT
# ============================================================

fig = go.Figure()

cone_style = {"0.99": ("rgba(255,165,0,0.10)", "99% risk cone"),
              "0.95": ("rgba(255,165,0,0.18)", "95% risk cone"),
              "0.68": ("rgba(255,165,0,0.30)", "68% risk cone")}
for q, (low, high) in cone.items():
    colour, name = cone_style[q]
    fig.add_trace(go.Scatter(x=cone_times, y=high, mode="lines", line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=cone_times, y=low, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=colour, name=name, hoverinfo="skip"))

for i, path in enumerate(scenarios):
    fig.add_trace(go.Scatter(x=cone_times, y=path, mode="lines",
                             line=dict(color="rgba(200,200,200,0.55)", width=1),
                             name="Simulated scenarios", legendgroup="scen", showlegend=(i == 0),
                             hoverinfo="skip"))

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

if realised is not None:
    fig.add_trace(go.Scatter(x=realised.index, y=realised.values, mode="lines",
                             line=dict(color="white", width=2, dash="dash"),
                             name="Realised (after the window)"))

fig.add_trace(go.Scatter(
    x=dates,
    y=prices,
    mode="lines+markers+text",
    line=dict(color="orange", width=3),
    marker=dict(size=9),
    text=["" if i < len(labels) - 1 else lab for i, lab in enumerate(labels)],
    hovertext=labels,
    textposition="top right",
    name="Forecast (CNN-BiLSTM)"
))

period = f"{df_h.index[0]:%B %Y} → {df_h.index[-1]:%d %B %Y}"

# Initial view: zoom on the observation window and the forecast (zoom out to see the full history)
x_start = df_window.index[0]
x_end = dates[-1] + pd.Timedelta(hours=max(2, horizon_sel // 4))
visible = [df_window["close"].values, np.array(prices)]
if cone:
    visible += list(cone["0.99"])
if len(scenarios):
    visible.append(np.ravel(scenarios))
if realised is not None:
    visible.append(realised.values)
visible = np.concatenate(visible)
pad = (visible.max() - visible.min()) * 0.08 or visible.max() * 0.01
y_low, y_high = visible.min() - pad, visible.max() + pad
fig.update_layout(
    title=dict(
        text=(
            "<b>CNN-BiLSTM FORECAST AND RISK CONE — BTC/USDT<br>"
            f"HOURLY CLOSES ({period})</b>"
        ),
        x=0.5,
        xanchor="center",
        font=dict(size=26)
    ),
    legend=dict(
        font=dict(size=16)
    ),
    hovermode="x unified",
    dragmode="zoom",
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font=dict(color="white"),
    xaxis=dict(showgrid=True, gridcolor="#333", range=[x_start, x_end]),
    yaxis=dict(showgrid=True, gridcolor="#333", range=[y_low, y_high]),
    height=850
)

st.plotly_chart(fig, width="stretch", theme=None, config={"scrollZoom": True})

if risk_cone:
    st.caption(
        f"Current hourly volatility (EWMA, λ = {EWMA_LAMBDA}): {sigma_now * 100:.2f}% · "
        "The orange line is the model's expected path: it is almost flat because the best forecast of a "
        "random-walk-like price is close to today's price. The cone shows how far the price can move; "
        "grey lines are simulated scenarios (not forecasts). Move the observation window into the past "
        "to compare the cone with what actually happened (white dashed line)."
    )

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

comparison = load_comparison()
if comparison:
    with st.expander("Comparison with a Temporal Fusion Transformer (TFT)", expanded=False):
        st.markdown(
            "A TFT (pytorch-forecasting) was trained on the same data, split and test period, "
            "with causal inputs (72 hours of returns, absolute returns, 24h volatility, hour and "
            f"weekday). Same {comparison['n_origins']:,} forecast origins ({comparison['period']})."
        )
        rows = []
        for h in comparison["cnn_bilstm"]:
            c, t = comparison["cnn_bilstm"][h], comparison["tft"][h]
            rows.append({
                "Horizon": h,
                "R² CNN-BiLSTM": round(c["r2_oos"], 4), "R² TFT": round(t["r2_oos"], 4),
                "Direction CNN-BiLSTM": f"{c['hit']:.1%}", "Direction TFT": f"{t['hit']:.1%}",
                "Diebold-Mariano p (CNN)": round(c["dm_p_model_better"], 2),
                "Diebold-Mariano p (TFT)": round(t["dm_p_model_better"], 2),
                "Sharpe net CNN": round(c["sharpe_net"], 1), "Sharpe net TFT": round(t["sharpe_net"], 1),
                "Sharpe buy & hold": round(comparison["buy_and_hold_sharpe"][h], 1),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            "Neither model beats the random walk (negative R², no significant Diebold-Mariano test). "
            "The TFT learned the 2018-2024 upward drift and forecasts a rise most of the time, which "
            "hurts it over the test period. Strategy: long/short on non-overlapping periods, net of 0.1% per trade."
        )

if risk_cone:
    with st.expander("Backtesting of the risk cone (Kupiec test)", expanded=False):
        st.markdown(
            f"Multipliers calibrated on {risk_cone['calibration_period']}, tested on "
            f"{risk_cone['test_period']}. A well-calibrated 95% cone should contain the realised price "
            "about 95% of the time. Kupiec p-values are computed on non-overlapping observations; "
            "p < 0.05 means the coverage is significantly wrong."
        )
        rows = []
        for h, row in risk_cone["backtest"].items():
            rows.append({
                "Horizon": h,
                "95% empirical: coverage": f"{row['empirical_95']['coverage']:.1%}",
                "95% empirical: Kupiec p": round(row["empirical_95"]["kupiec_p"], 2),
                "95% Gaussian: coverage": f"{row['gaussian_95']['coverage']:.1%}",
                "95% Gaussian: Kupiec p": round(row["gaussian_95"]["kupiec_p"], 2),
                "99% empirical: coverage": f"{row['empirical_99']['coverage']:.1%}",
                "99% empirical: Kupiec p": round(row["empirical_99"]["kupiec_p"], 2),
                "99% Gaussian: coverage": f"{row['gaussian_99']['coverage']:.1%}",
                "99% Gaussian: Kupiec p": round(row["gaussian_99"]["kupiec_p"], 2),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            "Bitcoin returns have fat tails: a Gaussian cone is too narrow and is rejected at every horizon. "
            "The empirical (fat-tailed) cone used in the chart passes the test, except a slight over-coverage "
            "at 8h for 95%, which is on the prudent side."
        )

