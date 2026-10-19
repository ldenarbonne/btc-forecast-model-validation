"""RMSE of all models vs the random walk, on the same test origins (July 2025 - Oct 2026)."""
import json, warnings
import numpy as np, pandas as pd, torch, joblib
from scipy import stats
from train import CNN_BiLSTM
from features import HORIZONS, SEQ_LEN
warnings.filterwarnings("ignore")
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
origins = None
for df in models.values():
    origins = df.index if origins is None else origins.intersection(df.index)
realised = pd.DataFrame({h: logp.shift(-h).loc[origins] - logp.loc[origins] for h in HORIZONS}).dropna()
origins = realised.index
price0 = s.loc[origins].values

def nw_var(x, L):
    x = x - x.mean(); n = len(x); v = x @ x / n
    for k in range(1, L + 1): v += 2 * (1 - k / (L + 1)) * (x[k:] @ x[:-k]) / n
    return v / n

out = {"period": f"{origins.min():%Y-%m-%d} -> {origins.max():%Y-%m-%d}", "n": len(origins), "random_walk": {}, "models": {}}
for h in HORIZONS:
    y = realised[h].values
    out["random_walk"][f"{h}h"] = {"rmse_pct": float(np.sqrt(np.mean(y ** 2)) * 100),
                                   "rmse_usd": float(np.sqrt(np.mean((price0 * (np.exp(y) - 1)) ** 2)))}
for name, F in models.items():
    rep = {}
    for h in HORIZONS:
        f, y = F.loc[origins, h].values, realised[h].values
        ae_m, ae_rw = (y - f) ** 2, y ** 2
        dd = ae_m - ae_rw
        rep[f"{h}h"] = {
            "rmse_pct": float(np.sqrt(ae_m.mean()) * 100),
            "rmse_usd": float(np.sqrt(np.mean((price0 * (np.exp(y) - np.exp(f))) ** 2))),
            "skill_vs_rw": float(1 - np.sqrt(ae_m.mean() / ae_rw.mean())),
            "dm_p_model_better": float(stats.norm.cdf(dd.mean() / np.sqrt(nw_var(dd, h + 5)))),
        }
    out["models"][name] = rep
json.dump(out, open("rmse_comparison.json", "w"), indent=1)
print(out["period"], out["n"])
for h in ["1h", "4h", "24h"]:
    rw = out["random_walk"][h]
    print(f"--- {h}: random walk RMSE {rw['rmse_pct']:.3f}% (${rw['rmse_usd']:.0f})")
    for name, rep in out["models"].items():
        x = rep[h]
        print(f"   {name:34s} RMSE {x['rmse_pct']:.4f}%  ${x['rmse_usd']:.0f}  skill {x['skill_vs_rw']*100:+.2f}%  DM p {x['dm_p_model_better']:.3f}")
