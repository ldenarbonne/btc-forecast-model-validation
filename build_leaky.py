"""Comparison only: VMD computed on whole yearly blocks (uses future data inside each block), as in the original app."""
import numpy as np, pandas as pd
from vmdpy import VMD
from features import SEQ_LEN, HORIZONS, VMD_PARAMS, incremental_targets
s = pd.read_csv("btc_hourly.csv", index_col=0, parse_dates=True)["close"]
logp = np.log(s.values); lr = np.diff(logp); lr_t = s.index[1:]
X, Y, T = [], [], []
for year in sorted(set(lr_t.year)):
    m = np.where(lr_t.year == year)[0]
    if len(m) % 2: m = m[1:]
    p = VMD_PARAMS
    u, _, _ = VMD(lr[m], p["alpha"], p["tau"], p["K"], p["DC"], p["init"], p["tol"]); imf = u.T.astype(np.float32)
    for j in range(SEQ_LEN - 1, len(m)):
        t = m[j] + 1                                     # hour index in s
        if t + max(HORIZONS) >= len(s): break
        X.append(imf[j - SEQ_LEN + 1: j + 1]); Y.append(incremental_targets(logp, t)); T.append(s.index[t])
    print(year, len(m), flush=True)
np.savez_compressed("dataset_leaky.npz", X=np.stack(X), y=np.stack(Y), times=pd.DatetimeIndex(T).values.astype("datetime64[s]").astype(np.int64))
