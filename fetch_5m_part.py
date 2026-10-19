import time, sys
import pandas as pd, requests
URL = "https://data-api.binance.vision/api/v3/klines"
start, end, out = sys.argv[1], sys.argv[2], sys.argv[3]
cursor = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000); stop = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
rows = []; sess = requests.Session()
while cursor < stop:
    for k in range(5):
        try:
            b = sess.get(URL, params={"symbol": "BTCUSDT", "interval": "5m", "startTime": cursor, "endTime": stop - 1, "limit": 1000}, timeout=30).json(); break
        except Exception:
            time.sleep(2)
    if not b: break
    rows += [(x[0], float(x[4]), x[6]) for x in b]
    if len(b) < 1000: break
    cursor = b[-1][0] + 300_000
pd.DataFrame(rows, columns=["t", "close", "close_time"]).to_csv(out, index=False)
print(out, len(rows))
