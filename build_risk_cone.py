"""
Risk cone for BTC/USDT: EWMA volatility (RiskMetrics, lambda = 0.94) scaled by empirical,
fat-tailed multipliers, calibrated on the training period and backtested on the test period.

For each horizon h (1..24 hours) and level q, the multiplier c(h, q) is the q-quantile of
|ln P(t+h) - ln P(t)| / (sigma_t * sqrt(h)) over the calibration period, where sigma_t is the
EWMA volatility of hourly log-returns known at time t. The band is P(t) * exp(+/- c * sigma_t * sqrt(h)).

Backtest on the test period: coverage and Kupiec unconditional-coverage test on non-overlapping
observations, for the empirical cone and for the Gaussian cone (c = normal quantile).
Also stores a sample of standardised hourly returns used by the app to simulate scenarios.
Output: risk_cone.json
"""
import argparse, json
import numpy as np, pandas as pd
from scipy import stats

LAMBDA = 0.94
LEVELS = [0.68, 0.95, 0.99]
H = list(range(1, 25))
REPORT_H = [1, 2, 4, 6, 8, 12, 24]


def kupiec_p(inside: np.ndarray, q: float) -> float:
    n, f = len(inside), int((~inside).sum())
    p = 1 - q
    if f == 0:
        lr = -2 * n * np.log(1 - p)
    elif f == n:
        lr = -2 * n * np.log(p)
    else:
        ph = f / n
        lr = -2 * ((n - f) * np.log(1 - p) + f * np.log(p) - ((n - f) * np.log(1 - ph) + f * np.log(ph)))
    return float(1 - stats.chi2.cdf(lr, 1))


ap = argparse.ArgumentParser()
ap.add_argument("--data", default="btc_hourly.csv")
ap.add_argument("--calib-end", default="2025-06-30")
ap.add_argument("--test-start", default="2025-07-01")
ap.add_argument("--out", default="risk_cone.json")
a = ap.parse_args()

s = pd.read_csv(a.data, index_col=0, parse_dates=True)["close"]
lp = np.log(s)
r = lp.diff().dropna()
sigma = np.sqrt(r.pow(2).ewm(alpha=1 - LAMBDA, adjust=False).mean())   # known at t (uses r up to t)

multipliers, backtest = {}, {}
for h in H:
    z = ((lp.shift(-h) - lp) / (sigma * np.sqrt(h))).dropna()
    z_cal, z_test = z[:a.calib_end].abs(), z[a.test_start:].abs()
    multipliers[str(h)] = {str(q): float(np.quantile(z_cal, q)) for q in LEVELS}
    if h in REPORT_H:
        row = {}
        for q in (0.95, 0.99):
            c_emp, c_gau = multipliers[str(h)][str(q)], float(stats.norm.ppf(0.5 + q / 2))
            for name, c in (("empirical", c_emp), ("gaussian", c_gau)):
                inside = (z_test <= c).values
                row[f"{name}_{int(q*100)}"] = {
                    "multiplier": c,
                    "coverage": float(inside.mean()),
                    "kupiec_p": kupiec_p(inside[::h], q),       # non-overlapping observations
                    "n_nonoverlap": int(len(inside[::h])),
                }
        backtest[f"{h}h"] = row

# standardised hourly returns of the calibration period, for scenario simulation
z1 = (r / sigma.shift(1)).dropna()[:a.calib_end]
rng = np.random.default_rng(0)
sample = rng.choice(z1.values, size=5000, replace=False)

out = {"lambda": LAMBDA, "calibration_period": f"{z1.index.min():%Y-%m-%d} -> {a.calib_end}",
       "test_period": f"{a.test_start} -> {s.index[-1]:%Y-%m-%d}",
       "levels": LEVELS, "multipliers": multipliers, "backtest": backtest,
       "standardised_hourly_returns": [round(float(x), 4) for x in sample]}
json.dump(out, open(a.out, "w"), indent=1)
for h, row in backtest.items():
    e, g = row["empirical_95"], row["gaussian_95"]
    e9, g9 = row["empirical_99"], row["gaussian_99"]
    print(f"{h:>4}  95%: emp {e['coverage']:.3f} (p={e['kupiec_p']:.2f})  gauss {g['coverage']:.3f} (p={g['kupiec_p']:.2f})   "
          f"99%: emp {e9['coverage']:.3f} (p={e9['kupiec_p']:.2f})  gauss {g9['coverage']:.3f} (p={g9['kupiec_p']:.2f})")
