"""
Same out-of-sample evaluation for: random walk, CNN-BiLSTM (causal VMD), TFT.
Common forecast origins only. Metrics: out-of-sample R2 vs random walk, directional accuracy,
Diebold-Mariano (Newey-West variance), Pesaran-Timmermann on non-overlapping samples,
and a long/short strategy on non-overlapping periods net of 0.1% per trade.
"""
import json
import numpy as np, pandas as pd, torch, joblib
from scipy import stats
from train import CNN_BiLSTM
from features import HORIZONS, SEQ_LEN

COST = 0.001
s = pd.read_csv("btc_hourly.csv", index_col=0, parse_dates=True)["close"]
logp = pd.Series(np.log(s.values), index=s.index)

# ---- CNN-BiLSTM forecasts
d = np.load("dataset.npz"); X = d["X"]; t = pd.to_datetime(d["times"], unit="s")
te = t >= pd.Timestamp("2025-07-01")
sc = joblib.load("models/1h_2h_4h_6h_8h_12h_24h.pkl")
m = CNN_BiLSTM(len(HORIZONS)); m.load_state_dict(torch.load("models/1h_2h_4h_6h_8h_12h_24h.pth")); m.eval()
with torch.no_grad():
    p = sc["y"].inverse_transform(m(torch.tensor(sc["X"].transform(X[te].reshape(-1, 3)).reshape(-1, SEQ_LEN, 3), dtype=torch.float32)).numpy())
cnn = pd.DataFrame(np.cumsum(p, 1), index=t[te], columns=HORIZONS)

# ---- TFT forecasts
tft_raw = pd.read_csv("tft_test_predictions.csv", parse_dates=["origin"]).set_index("origin")
tft = pd.DataFrame({h: tft_raw[f"h{h}"] for h in HORIZONS})

origins = cnn.index.intersection(tft.index)
realised = pd.DataFrame({h: logp.shift(-h).loc[origins] - logp.loc[origins] for h in HORIZONS}).dropna()
origins = realised.index


def nw_var(x, lags):
    x = x - x.mean(); n = len(x); v = x @ x / n
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (x[k:] @ x[:-k]) / n
    return v / n


def pt_stat(f, y):
    a, b = f > 0, y > 0; n = len(a)
    hit, py, px = (a == b).mean(), b.mean(), a.mean()
    ps = py * px + (1 - py) * (1 - px)
    v = ps * (1 - ps) / n - ((2 * py - 1) ** 2 * px * (1 - px) / n + (2 * px - 1) ** 2 * py * (1 - py) / n
                             + 4 * py * px * (1 - py) * (1 - px) / n ** 2)
    return (hit - ps) / np.sqrt(max(v, 1e-12))


def evaluate(fc):
    out = {}
    for h in HORIZONS:
        f, y = fc.loc[origins, h].values, realised[h].values
        e_m, e_rw = (y - f) ** 2, y ** 2
        dd = e_m - e_rw
        dm = dd.mean() / np.sqrt(nw_var(dd, h + 5))
        pts = [pt_stat(f[o::h], y[o::h]) for o in range(h)]
        pos = np.sign(f[::h]); r = y[::h]
        net = pos * r - np.abs(np.diff(np.concatenate([[0], pos]))) * COST
        out[f"{h}h"] = {
            "r2_oos": float(1 - e_m.sum() / e_rw.sum()),
            "hit": float(np.mean(np.sign(f) == np.sign(y))),
            "dm_p_model_better": float(stats.norm.cdf(dm)),
            "pt_p_nonoverlap": float(1 - stats.norm.cdf(np.mean(pts))),
            "sharpe_net": float(net.mean() / net.std() * np.sqrt(24 * 365 / h)) if net.std() > 0 else 0.0,
        }
    return out


bh = {f"{h}h": float(realised[h].values[::h].mean() / realised[h].values[::h].std() * np.sqrt(24 * 365 / h)) for h in HORIZONS}
up = {f"{h}h": float((realised[h] > 0).mean()) for h in HORIZONS}
report = {"n_origins": len(origins), "period": f"{origins.min():%Y-%m-%d} -> {origins.max():%Y-%m-%d}",
          "cnn_bilstm": evaluate(cnn), "tft": evaluate(tft), "buy_and_hold_sharpe": bh, "share_up": up}
json.dump(report, open("comparison.json", "w"), indent=2)

print(report["period"], report["n_origins"], "origins")
hdr = f"{'h':>4} | {'R2 CNN':>8} {'R2 TFT':>8} | {'hit CNN':>7} {'hit TFT':>7} {'up%':>5} | {'DM p CNN':>8} {'DM p TFT':>8} | {'PT p CNN':>8} {'PT p TFT':>8} | {'Shp CNN':>7} {'Shp TFT':>7} {'B&H':>6}"
print(hdr)
for h in [f"{x}h" for x in HORIZONS]:
    c, f = report["cnn_bilstm"][h], report["tft"][h]
    print(f"{h:>4} | {c['r2_oos']:8.4f} {f['r2_oos']:8.4f} | {c['hit']:7.3f} {f['hit']:7.3f} {up[h]:5.3f} | "
          f"{c['dm_p_model_better']:8.3f} {f['dm_p_model_better']:8.3f} | {c['pt_p_nonoverlap']:8.3f} {f['pt_p_nonoverlap']:8.3f} | "
          f"{c['sharpe_net']:7.2f} {f['sharpe_net']:7.2f} {bh[h]:6.2f}")
