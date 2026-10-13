"""
Train the CNN-BiLSTM on causal VMD features and evaluate it out of sample.

Chronological split (no shuffling across time):
  train : up to TRAIN_END
  valid : TRAIN_END -> VALID_END   (early stopping)
  test  : after VALID_END          (never seen during training or model selection)
Outputs: models/1h_2h_4h_6h_8h_12h_24h.pth, .pkl (scalers) and evaluation.json
"""
import json, argparse, time
import numpy as np, pandas as pd, torch, torch.nn as nn, joblib
from sklearn.preprocessing import StandardScaler
from scipy.stats import binomtest
from features import HORIZONS, SEQ_LEN, VMD_WINDOW


class CNN_BiLSTM(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 32, 3)
        self.conv2 = nn.Conv1d(32, 64, 3)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(64, 64, num_layers=2, bidirectional=True, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(128, out_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset.npz")
    ap.add_argument("--train-end", default="2024-07-01")
    ap.add_argument("--valid-end", default="2025-07-01")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--out-dir", default="models")
    a = ap.parse_args()
    torch.manual_seed(42); np.random.seed(42)

    d = np.load(a.data)
    X, y, t = d["X"], d["y"], pd.to_datetime(d["times"], unit="s")
    gap = max(HORIZONS)                      # purge: targets must not overlap the next period
    tr = t < pd.Timestamp(a.train_end) - pd.Timedelta(hours=gap)
    va = (t >= pd.Timestamp(a.train_end)) & (t < pd.Timestamp(a.valid_end) - pd.Timedelta(hours=gap))
    te = t >= pd.Timestamp(a.valid_end)

    sx = StandardScaler().fit(X[tr].reshape(-1, 3))
    sy = StandardScaler().fit(y[tr])
    f = lambda m: torch.tensor(sx.transform(X[m].reshape(-1, 3)).reshape(-1, SEQ_LEN, 3), dtype=torch.float32)
    Xtr, Xva, Xte = f(tr), f(va), f(te)
    ytr = torch.tensor(sy.transform(y[tr]), dtype=torch.float32)
    yva = torch.tensor(sy.transform(y[va]), dtype=torch.float32)

    model = CNN_BiLSTM(len(HORIZONS))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    best, best_state, bad = np.inf, None, 0
    for ep in range(a.epochs):
        t0 = time.time(); model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            opt.zero_grad(); loss = loss_fn(model(Xtr[b]), ytr[b]); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xva), yva).item()
        print(f"epoch {ep+1:2d}  valid MSE {vl:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience: break
    model.load_state_dict(best_state); model.eval()

    # ---------- out-of-sample evaluation (test period) ----------
    with torch.no_grad():
        pred = sy.inverse_transform(model(Xte).numpy())
    cum_pred, cum_true = np.cumsum(pred, axis=1), np.cumsum(y[te], axis=1)   # log-return from t to t+h
    report = {"train_period": f"{t[tr].min():%Y-%m-%d} -> {t[tr].max():%Y-%m-%d}",
              "valid_period": f"{t[va].min():%Y-%m-%d} -> {t[va].max():%Y-%m-%d}",
              "test_period": f"{t[te].min():%Y-%m-%d} -> {t[te].max():%Y-%m-%d}",
              "n_train": int(tr.sum()), "n_valid": int(va.sum()), "n_test": int(te.sum()),
              "vmd_window_hours": VMD_WINDOW, "horizons": {}}
    for k, h in enumerate(HORIZONS):
        p_, y_ = cum_pred[:, k], cum_true[:, k]
        r2 = 1 - np.sum((y_ - p_) ** 2) / np.sum(y_ ** 2)          # vs random walk (zero return)
        hits = int(np.sum(np.sign(p_) == np.sign(y_))); n = len(y_)
        up_rate = float(np.mean(y_ > 0))
        report["horizons"][f"{h}h"] = {
            "rmse_model_pct": float(np.sqrt(np.mean((y_ - p_) ** 2)) * 100),
            "rmse_random_walk_pct": float(np.sqrt(np.mean(y_ ** 2)) * 100),
            "r2_oos_vs_random_walk": float(r2),
            "directional_accuracy": hits / n,
            "share_of_up_moves": up_rate,
            "p_value_vs_coin_flip": float(binomtest(hits, n, 0.5, alternative="greater").pvalue),
        }
    import os; os.makedirs(a.out_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{a.out_dir}/1h_2h_4h_6h_8h_12h_24h.pth")
    joblib.dump({"X": sx, "y": sy}, f"{a.out_dir}/1h_2h_4h_6h_8h_12h_24h.pkl")
    json.dump(report, open(f"{a.out_dir}/evaluation.json", "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
