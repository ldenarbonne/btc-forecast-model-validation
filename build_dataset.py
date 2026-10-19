"""Build (X, y) samples with causal VMD features. Uses all CPU cores."""
import time, argparse
from multiprocessing import Pool
import numpy as np, pandas as pd
from features import VMD_WINDOW, HORIZONS, vmd_window, incremental_targets

def _work(args):
    lr, idx = args
    return [vmd_window(lr[t - VMD_WINDOW + 1: t + 1]) for t in idx]

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--data", default="btc_hourly.csv"); p.add_argument("--out", default="dataset.npz")
    a = p.parse_args()
    s = pd.read_csv(a.data, index_col=0, parse_dates=True)["close"]
    logp = np.log(s.values)
    lr = np.diff(logp)                          # lr[i] = return from hour i to i+1 (ends at hour i+1)
    # sample at hour t (index in s) uses returns ending at t -> lr index t-1
    ts = np.arange(VMD_WINDOW, len(s) - max(HORIZONS))
    lr_idx = ts - 1
    chunks = np.array_split(lr_idx, 64)
    t0 = time.time()
    with Pool() as pool:
        res = pool.map(_work, [(lr, c) for c in chunks])
    X = np.concatenate([np.stack(r) for r in res])
    y = np.stack([incremental_targets(logp, t) for t in ts])
    np.savez_compressed(a.out, X=X, y=y, times=s.index[ts].values.astype("datetime64[s]").astype(np.int64))
    print(f"{len(X):,} samples, X{X.shape}, y{y.shape}, {time.time()-t0:.0f}s -> {a.out}")
