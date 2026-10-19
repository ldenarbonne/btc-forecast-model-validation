"""
Temporal Fusion Transformer (pytorch-forecasting) trained on the same Binance hourly
BTC/USDT data and the same chronological split as the CNN-BiLSTM, for a fair comparison.

Inputs (all causal): past hourly log-returns, absolute returns, 24h rolling volatility,
plus known calendar features (hour of day, day of week). Encoder: 72 hours.
Output: the next 24 hourly log-returns (median of the quantile forecast).
Writes tft_test_predictions.csv: forecast origin + cumulative forecasts for 1..24h.
"""
import argparse, warnings
import numpy as np, pandas as pd, torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import TorchNormalizer
from pytorch_forecasting.metrics import QuantileLoss

warnings.filterwarnings("ignore")
ENC, PRED = 72, 24

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="btc_hourly.csv")
ap.add_argument("--train-end", default="2024-07-01")
ap.add_argument("--valid-end", default="2025-07-01")
ap.add_argument("--epochs", type=int, default=10)
a = ap.parse_args()
pl.seed_everything(42)
torch.set_num_threads(2)

s = pd.read_csv(a.data, index_col=0, parse_dates=True)["close"]
df = pd.DataFrame({"time": s.index, "logp": np.log(s.values)})
df["ret"] = df["logp"].diff() * 100                       # % log-return of the hour ending at 'time'
df = df.dropna().reset_index(drop=True)
df["abs_ret"] = df["ret"].abs()
df["vol24"] = df["ret"].rolling(24).std()
df = df.dropna().reset_index(drop=True)
df["time_idx"] = np.arange(len(df))
df["hour"] = df["time"].dt.hour.astype(str)
df["dow"] = df["time"].dt.dayofweek.astype(str)
df["series"] = "BTC"

idx_of = lambda ts: int(df.index[df["time"] >= pd.Timestamp(ts)][0])
train_end, valid_end = idx_of(a.train_end), idx_of(a.valid_end)

common = dict(time_idx="time_idx", target="ret", group_ids=["series"],
              max_encoder_length=ENC, max_prediction_length=PRED,
              min_encoder_length=ENC, min_prediction_length=PRED,
              time_varying_known_categoricals=["hour", "dow"],
              time_varying_unknown_reals=["ret", "abs_ret", "vol24"],
              target_normalizer=TorchNormalizer(method="standard"),
              add_relative_time_idx=True)
training = TimeSeriesDataSet(df[df.time_idx < train_end - PRED], **common)
validation = TimeSeriesDataSet.from_dataset(training, df[df.time_idx < valid_end - PRED],
                                            min_prediction_idx=train_end, stop_randomization=True)
test = TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=valid_end + 1, stop_randomization=True)

tl = training.to_dataloader(train=True, batch_size=256, num_workers=0)
vl = validation.to_dataloader(train=False, batch_size=1024, num_workers=0)
te = test.to_dataloader(train=False, batch_size=1024, num_workers=0)

tft = TemporalFusionTransformer.from_dataset(training, learning_rate=1e-3, hidden_size=16,
        attention_head_size=2, dropout=0.1, hidden_continuous_size=8, loss=QuantileLoss())
trainer = pl.Trainer(max_epochs=a.epochs, accelerator="cpu", gradient_clip_val=0.1,
                     callbacks=[EarlyStopping(monitor="val_loss", patience=3)],
                     enable_progress_bar=False, logger=False, enable_checkpointing=True,
                     limit_train_batches=150)
trainer.fit(tft, tl, vl)
best = TemporalFusionTransformer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

out = best.predict(te, mode="prediction", return_index=True)
pred = out.output.cpu().numpy() / 100                  # hourly log-returns, shape (n, 24)
first = out.index["time_idx"].values                    # time_idx of the first forecast hour
origin_time = df.set_index("time_idx").loc[first - 1, "time"].values
cum = np.cumsum(pred, axis=1)
res = pd.DataFrame(cum, columns=[f"h{h}" for h in range(1, PRED + 1)])
res.insert(0, "origin", origin_time)
res.to_csv("tft_test_predictions.csv", index=False)
best_ckpt = trainer.checkpoint_callback.best_model_path
print(f"{len(res):,} test forecasts from {res.origin.min()} to {res.origin.max()}; best checkpoint {best_ckpt}")
