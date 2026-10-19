"""
HAR model of Corsi (2009) for Bitcoin realised variance, and benchmarks.
  RV_{t+1:t+h} = b0 + bd RV_t + bw RV_t^(w) + bm RV_t^(m)
with daily, weekly (7-day) and monthly (30-day) averages (crypto trades every day).
Models: HAR-RV (levels), log-HAR (logs, with lognormal bias correction), random walk RV,
EWMA RiskMetrics (lambda 0.94) on daily squared returns, EWMA on RV, GARCH(1,1) on daily returns.
Parameters estimated on the training period only (2019 -> 2024-06-30) and kept fixed.
Test: 2025-07-01 onward. Losses: QLIKE (robust to noise in RV, Patton 2011) and MSE;
Mincer-Zarnowitz R2; Diebold-Mariano (Newey-West) vs EWMA RiskMetrics.
Output: har_evaluation.json, har_params.json
"""
import json, warnings
import numpy as np, pandas as pd
from scipy import stats
from arch import arch_model
warnings.filterwarnings("ignore")
TRAIN_END, TEST_START = "2024-06-30", "2025-07-01"
d = pd.read_csv("btc_daily_rv.csv", index_col=0, parse_dates=True).asfreq("D")
d["rv"] = d["rv"].interpolate(); d["ret"] = d["ret"].fillna(0.0)
rv = d["rv"]
X = pd.DataFrame({"d": rv, "w": rv.rolling(7).mean(), "m": rv.rolling(30).mean()})

def target(h):  # average RV over the next h days
    return rv[::-1].rolling(h).mean()[::-1].shift(-1)

def ols(Xm, y):
    A = np.column_stack([np.ones(len(Xm)), Xm]); b, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ b; return b, res

def nw_var(x, L):
    x = x - x.mean(); n = len(x); v = x @ x / n
    for k in range(1, L + 1): v += 2 * (1 - k / (L + 1)) * (x[k:] @ x[:-k]) / n
    return v / n

def qlike(y, f): return y / f - np.log(y / f) - 1

# GARCH(1,1) on daily % returns, Student-t, fitted on training, filtered over full sample
am = arch_model(d["ret"][:TRAIN_END] * 100, vol="GARCH", p=1, q=1, dist="t", mean="Constant")
g = am.fit(disp="off"); om, a1, b1 = g.params["omega"], g.params["alpha[1]"], g.params["beta[1]"]; mu = g.params["mu"]
e = d["ret"].values * 100 - mu; s2 = np.empty(len(e)); s2[0] = om / (1 - a1 - b1)
for t in range(1, len(e)): s2[t] = om + a1 * e[t - 1] ** 2 + b1 * s2[t - 1]
s2_next = om + a1 * e ** 2 + b1 * s2              # one-step-ahead variance for day t+1, known at t
garch1 = pd.Series(s2_next / 1e4, index=d.index)
pers = a1 + b1; uncond = om / (1 - pers)
def garch_h(h):  # average variance over next h days
    k = np.arange(h); out = []
    for v in s2_next:
        out.append(np.mean(uncond + pers ** k * (v - uncond)))
    return pd.Series(np.array(out) / 1e4, index=d.index)

ewma_r = (d["ret"] ** 2).ewm(alpha=0.06, adjust=False).mean()
ewma_rv = rv.ewm(alpha=0.06, adjust=False).mean()

params, report = {}, {"train": f"{X.dropna().index.min():%Y-%m-%d} -> {TRAIN_END}", "test_start": TEST_START, "horizons": {}}
for h in (1, 7):
    y = target(h)
    ok = X.notna().all(1) & y.notna()
    tr = ok & (X.index <= TRAIN_END)
    b, _ = ols(X[tr].values, y[tr].values)
    bl, res_l = ols(np.log(X[tr].values), np.log(y[tr].values))
    params[f"{h}d"] = {"har": b.tolist(), "log_har": bl.tolist(), "log_har_resid_var": float(res_l.var())}
    A = np.column_stack([np.ones(ok.sum()), X[ok].values])
    f = {
        "HAR-RV": pd.Series(A @ b, index=X[ok].index).clip(lower=1e-8),
        "log-HAR": pd.Series(np.exp(np.column_stack([np.ones(ok.sum()), np.log(X[ok].values)]) @ bl + res_l.var() / 2), index=X[ok].index),
        "Random walk (RV today)": rv[ok] if h == 1 else X["w"][ok],
        "EWMA RiskMetrics (daily returns)": ewma_r[ok],
        "EWMA on realised variance": ewma_rv[ok],
        "GARCH(1,1) Student-t": (garch1 if h == 1 else garch_h(h))[ok],
    }
    te = y[ok].index >= TEST_START
    yt = y[ok][te].values
    rows = {}
    base = qlike(yt, f["EWMA RiskMetrics (daily returns)"][te].values)
    for name, fc in f.items():
        ft = fc[te].values
        q = qlike(yt, ft); mse = (yt - ft) ** 2
        Amz = np.column_stack([np.ones(len(ft)), ft]); bmz, *_ = np.linalg.lstsq(Amz, yt, rcond=None)
        r2 = 1 - ((yt - Amz @ bmz) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
        dd = q - base
        p = float(stats.norm.cdf(dd.mean() / np.sqrt(nw_var(dd, h + 5)))) if name != "EWMA RiskMetrics (daily returns)" else None
        rows[name] = {"qlike": float(q.mean()), "rmse_vol_pts": float(np.sqrt(np.mean((np.sqrt(365 * yt) - np.sqrt(365 * ft)) ** 2)) * 100),
                      "mz_r2": float(r2), "dm_p_vs_ewma": p}
    report["horizons"][f"{h}d"] = {"n_test_days": int(te.sum()), "models": rows}
    print(f"\n=== horizon {h} day(s), test days {te.sum()} ; HAR coefs b0,bd,bw,bm = {np.round(b,6)}")
    for name, x in rows.items():
        print(f"{name:34s} QLIKE {x['qlike']:.4f}  RMSE(ann. vol, pts) {x['rmse_vol_pts']:5.2f}  MZ R2 {x['mz_r2']:.3f}  DM p vs EWMA {x['dm_p_vs_ewma']}")
params["garch"] = {"omega": float(om), "alpha": float(a1), "beta": float(b1), "mu": float(mu), "nu": float(g.params["nu"])}
json.dump(report, open("har_evaluation.json", "w"), indent=1); json.dump(params, open("har_params.json", "w"), indent=1)
