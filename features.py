"""
Causal VMD features: for each hour t, the VMD is computed ONLY on the
VMD_WINDOW hourly log-returns ending at t (no future data), and the last
SEQ_LEN rows of the 3 modes are kept as the model input.
"""
import numpy as np
from vmdpy import VMD

VMD_WINDOW = 256          # hours of history fed to each VMD (even number)
SEQ_LEN = 24              # hours of modes fed to the network
VMD_PARAMS = dict(alpha=2000, tau=0, K=3, DC=0, init=1, tol=1e-7)
HORIZONS = [1, 2, 4, 6, 8, 12, 24]


def vmd_window(log_returns_window: np.ndarray) -> np.ndarray:
    """Return the last SEQ_LEN rows of the 3 VMD modes, shape (SEQ_LEN, 3)."""
    p = VMD_PARAMS
    u, _, _ = VMD(log_returns_window.astype(np.float64), p["alpha"], p["tau"], p["K"], p["DC"], p["init"], p["tol"])
    return u.T[-SEQ_LEN:].astype(np.float32)


def incremental_targets(log_price: np.ndarray, t: int) -> np.ndarray:
    """Log-returns between successive horizons: ln P(t+h_k) - ln P(t+h_{k-1}), with h_0 = 0."""
    prev, out = 0, []
    for h in HORIZONS:
        out.append(log_price[t + h] - log_price[t + prev]); prev = h
    return np.array(out, dtype=np.float32)
