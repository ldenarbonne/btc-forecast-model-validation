"""
Discriminatory power for the DIRECTION of the move (up = 1, down = 0), using each model's
forecast return as the score: ROC AUC, Gini = Accuracy Ratio = 2*AUC - 1.
95% CI by block bootstrap (one-week blocks), as forecasts at successive hours are dependent.
"""
import json, warnings
import numpy as np, pandas as pd, torch, joblib
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
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
rng = np.random.default_rng(0)
out = {"period": f"{origins.min():%Y-%m-%d} -> {origins.max():%Y-%m-%d}", "models": {}}
for name, F in models.items():
    rep = {}
    for h in HORIZONS:
        score, up = F.loc[origins, h].values, (realised[h].values > 0).astype(int)
        auc = roc_auc_score(up, score)
        N, B = len(up), 168; nb = N // B; boots = []
        for _ in range(500):
            ix = np.concatenate([np.arange(a, a + B) for a in rng.integers(0, N - B, nb)])
            if up[ix].min() != up[ix].max(): boots.append(roc_auc_score(up[ix], score[ix]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rep[f"{h}h"] = {"auc": float(auc), "auc_ci": [float(lo), float(hi)], "gini": float(2 * auc - 1),
                        "gini_ci": [float(2 * lo - 1), float(2 * hi - 1)]}
    out["models"][name] = rep
json.dump(out, open("roc_comparison.json", "w"), indent=1)
for h in ["1h", "4h", "24h"]:
    print("---", h)
    for name, rep in out["models"].items():
        x = rep[h]; print(f"   {name:34s} AUC {x['auc']:.3f} [{x['auc_ci'][0]:.3f},{x['auc_ci'][1]:.3f}]  Gini/AR {x['gini']*100:+.1f}% [{x['gini_ci'][0]*100:+.1f}%,{x['gini_ci'][1]*100:+.1f}%]")
# ROC curves
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), dpi=160)
cols = {"CNN-BiLSTM + VMD": "#1f5f8b", "TFT": "#c0504d", "AR(2) = ARIMA(2,1,0)": "#2e8b57"}
for ax, h in zip(axes, [1, 24]):
    up = (realised[h].values > 0).astype(int)
    for name, c in cols.items():
        fpr, tpr, _ = roc_curve(up, models[name].loc[origins, h].values)
        ax.plot(fpr, tpr, color=c, lw=1.4, label=f"{name.split(' =')[0]} (Gini {out['models'][name][f'{h}h']['gini']*100:+.1f}%)")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random (Gini 0%)")
    ax.set_title(f"ROC curve — direction at {h}h"); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(fontsize=7, frameon=False, loc="lower right"); ax.set_aspect("equal")
plt.tight_layout(); plt.savefig("roc_curves.png")
