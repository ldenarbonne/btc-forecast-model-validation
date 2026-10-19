"""
Classical time-series baselines on the same data and test period:
 - AR(p) on hourly log-returns  (= ARIMA(p,1,0) on log-prices), p chosen by BIC on the training set
 - ARMA(1,1) on returns          (= ARIMA(1,1,1) on log-prices), estimated with statsmodels
 - Seasonal AR with daily lags 24 and 48 (SARIMA(p,1,0)(2,0,0)_24 on log-prices, additive approximation)
Parameters are estimated on the training period only, then fixed; forecasts for 1..24h are made
recursively from every hourly origin of the test period (no look-ahead).
"""
import json, warnings
import numpy as np, pandas as pd
from statsmodels.tsa.arima.model import ARIMA
warnings.filterwarnings("ignore")
HOR = [1, 2, 4, 6, 8, 12, 24]
s = pd.read_csv("btc_hourly.csv", index_col=0, parse_dates=True)["close"]
lp = np.log(s); r = lp.diff().dropna()
train = r[:"2024-06-29"]; idx_test = r.index[r.index >= "2025-07-01"]

def ols_ar(x, lags):
    L = max(lags); Y = x[L:]; Xm = np.column_stack([np.ones(len(Y))] + [x[L - l:-l] for l in lags])
    b, *_ = np.linalg.lstsq(Xm, Y, rcond=None); res = Y - Xm @ b
    n, k = len(Y), Xm.shape[1]; bic = n * np.log(res @ res / n) + k * np.log(n)
    return b, bic

def ar_forecasts(x_all, b, lags, origins_pos, H=24):
    L = max(lags); out = np.zeros((len(origins_pos), H))
    for i, pos in enumerate(origins_pos):
        hist = list(x_all[pos - L + 1: pos + 1])
        for h in range(H):
            f = b[0] + sum(b[j + 1] * hist[-l] for j, l in enumerate(lags))
            hist.append(f); out[i, h] = f
    return np.cumsum(out, 1)

x = r.values; pos_test = np.searchsorted(r.index, idx_test)
pos_test = pos_test[pos_test + 24 < len(x)]
origins = r.index[pos_test]
results = {}
# AR(p), p by BIC
bics = {p: ols_ar(train.values, list(range(1, p + 1)))[1] for p in range(1, 13)}
p_best = min(bics, key=bics.get); b, _ = ols_ar(train.values, list(range(1, p_best + 1)))
results[f"AR({p_best}) = ARIMA({p_best},1,0)"] = ar_forecasts(x, b, list(range(1, p_best + 1)), pos_test)
print("BIC-selected p:", p_best, "coefs:", np.round(b, 4))
# seasonal AR
lags_s = list(range(1, p_best + 1)) + [24, 48]
bs, _ = ols_ar(train.values, lags_s)
results[f"SARIMA({p_best},1,0)(2,0,0)24"] = ar_forecasts(x, bs, lags_s, pos_test)
print("seasonal coefs (lag24, lag48):", np.round(bs[-2:], 4))
# ARMA(1,1) via statsmodels on training, then fixed-parameter filtering
m = ARIMA(train.values, order=(1, 0, 1), trend="c").fit()
c, phi, theta = m.params[0], m.params[1], m.params[2]
mu = c  # statsmodels 'const' is the mean
eps = np.zeros(len(x)); 
for t_ in range(1, len(x)):
    eps[t_] = x[t_] - (mu + phi * (x[t_ - 1] - mu) + theta * eps[t_ - 1])
F = np.zeros((len(pos_test), 24))
for i, pos in enumerate(pos_test):
    f1 = mu + phi * (x[pos] - mu) + theta * eps[pos]
    F[i, 0] = f1
    for h in range(1, 24):
        F[i, h] = mu + phi * (F[i, h - 1] - mu)
results["ARMA(1,1) = ARIMA(1,1,1)"] = np.cumsum(F, 1)
print("ARMA(1,1) params mu, phi, theta:", np.round([mu, phi, theta], 5))
pd.to_pickle({"origins": origins, "forecasts": results}, "arima_forecasts.pkl")
