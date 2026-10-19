"""Same out-of-sample evaluation for all models (common forecast origins)."""
import json, warnings
import numpy as np, pandas as pd, torch, joblib
from scipy import stats
from train import CNN_BiLSTM
from features import HORIZONS, SEQ_LEN
warnings.filterwarnings("ignore")
COST = 0.001
s = pd.read_csv("btc_hourly.csv", index_col=0, parse_dates=True)["close"]; logp = np.log(s)
d = np.load("dataset.npz"); X = d["X"]; t = pd.to_datetime(d["times"], unit="s"); te = t >= pd.Timestamp("2025-07-01")
sc = joblib.load("models/1h_2h_4h_6h_8h_12h_24h.pkl"); m = CNN_BiLSTM(7)
m.load_state_dict(torch.load("models/1h_2h_4h_6h_8h_12h_24h.pth")); m.eval()
with torch.no_grad():
    p = sc["y"].inverse_transform(m(torch.tensor(sc["X"].transform(X[te].reshape(-1, 3)).reshape(-1, SEQ_LEN, 3), dtype=torch.float32)).numpy())
models = {"CNN-BiLSTM + VMD": pd.DataFrame(np.cumsum(p, 1), index=t[te], columns=HORIZONS)}
tft = pd.read_csv("tft_test_predictions.csv", parse_dates=["origin"]).set_index("origin")
models["TFT"] = pd.DataFrame({h: tft[f"h{h}"] for h in HORIZONS})
ar = pd.read_pickle("arima_forecasts.pkl")
for name, F in ar["forecasts"].items():
    models[name] = pd.DataFrame(F[:, [h - 1 for h in HORIZONS]], index=ar["origins"], columns=HORIZONS)
r_train = logp.diff()["2018-01-01":"2024-06-29"].mean()
origins = None
for df in models.values():
    origins = df.index if origins is None else origins.intersection(df.index)
realised = pd.DataFrame({h: logp.shift(-h).loc[origins] - logp.loc[origins] for h in HORIZONS}).dropna()
origins = realised.index
models["Constant drift (training mean)"] = pd.DataFrame({h: np.full(len(origins), r_train * h) for h in HORIZONS}, index=origins)

def nw_var(x, L):
    x = x - x.mean(); n = len(x); v = x @ x / n
    for k in range(1, L + 1): v += 2 * (1 - k / (L + 1)) * (x[k:] @ x[:-k]) / n
    return v / n
def wilson(k, n, z=1.96):
    ph = k / n; den = 1 + z * z / n; c = (ph + z * z / (2 * n)) / den
    hw = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den; return c - hw, c + hw
rng = np.random.default_rng(0)
report = {"period": f"{origins.min():%Y-%m-%d} -> {origins.max():%Y-%m-%d}", "n": len(origins), "models": {}}
for name, F in models.items():
    rep = {}
    for h in HORIZONS:
        f, y = F.loc[origins, h].values, realised[h].values
        em, e0 = (y - f) ** 2, y ** 2; dd = em - e0
        dm_p = float(stats.norm.cdf(dd.mean() / np.sqrt(nw_var(dd, h + 5))))
        N, B = len(e0), 168; nb = N // B
        boots = []
        for _ in range(500):
            ix = np.concatenate([np.arange(a, a + B) for a in rng.integers(0, N - B, nb)])
            boots.append(1 - em[ix].sum() / e0[ix].sum())
        fn, yn = f[::h], y[::h]; k = int((np.sign(fn) == np.sign(yn)).sum()); lo, hi = wilson(k, len(fn))
        pos = np.sign(fn); net = pos * yn - np.abs(np.diff(np.concatenate([[0], pos]))) * COST
        rep[f"{h}h"] = {"r2": float(1 - em.sum() / e0.sum()), "r2_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                        "hit": k / len(fn), "hit_ci": [lo, hi], "dm_p": dm_p,
                        "sharpe_net": float(net.mean() / net.std() * np.sqrt(8760 / h)) if net.std() > 0 else 0.0}
    report["models"][name] = rep
json.dump(report, open("comparison_all.json", "w"), indent=1)
print(report["period"], report["n"])
for h in ["1h", "4h", "24h"]:
    print(f"--- {h}")
    for name, rep in report["models"].items():
        x = rep[h]
        print(f"{name:38s} R2 {x['r2']:+.4f} [{x['r2_ci'][0]:+.4f},{x['r2_ci'][1]:+.4f}]  hit {x['hit']:.3f} [{x['hit_ci'][0]:.3f},{x['hit_ci'][1]:.3f}]  DM p {x['dm_p']:.2f}  Sharpe {x['sharpe_net']:+.1f}")
