"""Download hourly BTC/USDT candles from Binance's public market-data API (no API key)."""
import time, argparse
import pandas as pd, requests

ENDPOINTS = ["https://data-api.binance.vision/api/v3/klines", "https://api.binance.com/api/v3/klines"]

def fetch_hourly(start="2018-01-01", symbol="BTCUSDT"):
    cursor = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    rows = []
    url = ENDPOINTS[0]
    while True:
        for attempt in range(3):
            try:
                r = requests.get(url, params={"symbol": symbol, "interval": "1h", "startTime": cursor, "limit": 1000}, timeout=30)
                r.raise_for_status(); batch = r.json(); break
            except requests.RequestException:
                url = ENDPOINTS[(ENDPOINTS.index(url) + 1) % len(ENDPOINTS)]; time.sleep(2)
        else:
            raise RuntimeError("Binance unreachable")
        if not batch: break
        rows += batch
        if len(batch) < 1000: break
        cursor = batch[-1][0] + 3_600_000
        time.sleep(0.1)
    df = pd.DataFrame(rows).iloc[:, [0, 4, 6]]
    df.columns = ["open_time", "close", "close_time"]
    df = df[df.close_time < int(time.time() * 1000)]           # completed hours only
    s = pd.Series(df.close.astype(float).values, index=pd.to_datetime(df.open_time, unit="ms"), name="close")
    s = s[~s.index.duplicated()].sort_index().asfreq("1h").ffill()
    return s

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--start", default="2018-01-01"); p.add_argument("--out", default="btc_hourly.csv")
    a = p.parse_args()
    s = fetch_hourly(a.start); s.to_csv(a.out, index_label="datetime")
    print(f"{len(s):,} hourly closes from {s.index[0]} to {s.index[-1]} -> {a.out}")
