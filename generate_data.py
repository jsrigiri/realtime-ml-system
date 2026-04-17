import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

n = 3000
timestamps = pd.date_range("2025-01-01 09:30:00", periods=n, freq="s")

mid = np.zeros(n)
mid[0] = 100.0

for i in range(1, n):
    shock = np.random.normal(0, 0.03)
    regime = 0.01 * np.sin(i / 100)
    mid[i] = mid[i - 1] + shock + regime

spread = np.random.uniform(0.01, 0.03, n)
bid = mid - spread / 2
ask = mid + spread / 2
volume = np.random.randint(1, 20, n)

df = pd.DataFrame({
    "timestamp": timestamps,
    "mid": mid,
    "bid": bid,
    "ask": ask,
    "volume": volume,
})

Path("data").mkdir(exist_ok=True)
df.to_csv("data/historical_ticks.csv", index=False)
print(f"Saved {len(df)} rows to data/historical_ticks.csv")