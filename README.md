# Retraining the model

The model is trained on hourly BTC/USDT closes from Binance (2018 → today), with a
**causal** VMD: each sample decomposes only the 256 hours of log-returns that end at
the forecast time, so no future information leaks into the features.

```bash
cd training
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pandas numpy requests vmdpy scikit-learn scipy joblib

python fetch_data.py                 # ~77,000 hourly closes  -> btc_hourly.csv   (~1 min)
python build_dataset.py              # causal VMD features    -> dataset.npz      (~5-15 min)
python train.py --out-dir ../models  # model + scalers + evaluation.json          (~5-15 min)
```

Chronological split: train to 30 June 2024, validation (early stopping) to 30 June 2025,
test afterwards. Test results are written to `models/evaluation.json` and shown in the app.

Targets: log-returns between successive horizons (0→1h, 1h→2h, 2h→4h, … 12h→24h);
the app chains them to draw the forecast path.

## Why the VMD must be causal

Same network, same data, same test period (1 July 2025 → 12 October 2026, 11,000+ hourly forecasts).
Only the way the VMD is computed changes. `build_leaky.py` reproduces the original approach
(VMD over whole blocks of data, so each mode also contains information from later hours).

| Horizon | R² vs random walk — causal VMD | Direction correct — causal | R² — VMD over whole blocks (look-ahead) | Direction correct — look-ahead |
|---|---|---|---|---|
| 1h | -0.0005 | 50.3% | 0.62 | 75.8% |
| 2h | -0.0021 | 50.6% | 0.71 | 79.5% |
| 4h | -0.0021 | 49.9% | 0.62 | 76.5% |
| 6h | -0.0023 | 49.3% | 0.42 | 71.4% |
| 8h | -0.0020 | 50.6% | 0.21 | 66.6% |
| 12h | -0.0007 | 51.1% | 0.00 | 58.6% |
| 24h | -0.0006 | 51.5% | -0.05 | 56.2% |

With look-ahead, the model seems to explain up to 70% of hourly moves and to get the direction right
8 times out of 10. Once the future is removed, its forecasts are no better than a random walk.
The apparent skill came entirely from the decomposition seeing future prices.
