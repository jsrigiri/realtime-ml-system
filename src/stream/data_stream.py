import time
import pandas as pd
from config import STREAM_SLEEP_SEC


def stream_ticks(path: str):
    df = pd.read_csv(path, parse_dates=["timestamp"])

    for _, row in df.iterrows():
        tick = {
            "timestamp": row["timestamp"],
            "mid": float(row["mid"]),
            "bid": float(row["bid"]),
            "ask": float(row["ask"]),
            "volume": float(row["volume"]),
        }
        yield tick
        if STREAM_SLEEP_SEC > 0:
            time.sleep(STREAM_SLEEP_SEC)