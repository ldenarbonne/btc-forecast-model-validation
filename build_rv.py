"""Daily realised variance (UTC days) from 5-minute BTC/USDT log-returns."""
import glob, time
import numpy as np, pandas as pd
df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("p5m/*.csv"))])
df = df[df.close_time < time.time() * 1000].drop_duplicates("t").sort_values("t")
s = pd.Series(df.close.values, index=pd.to_datetime(df.t, unit="ms"))
r = np.log(s).diff().dropna()
day = r.index.floor("D")
rv = (r ** 2).groupby(day).sum()
n = r.groupby(day).size()
ret_d = r.groupby(day).sum()
out = pd.DataFrame({"rv": rv, "n_obs": n, "ret": ret_d})
out = out[out.n_obs >= 250]           # complete days only (288 expected)
out.index.name = "date"
out.to_csv("btc_daily_rv.csv")
print(len(out), out.index.min().date(), out.index.max().date(), "dropped days:", int((n < 250).sum()))
print("annualised vol (sqrt(365*RV)) median %:", round(np.sqrt(365 * out.rv).median() * 100, 1))
