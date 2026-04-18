import time
import numpy as np


class MetricsTracker:
    def __init__(self):
        self.prediction_latencies = []
        self.predictions = []
        self.targets = []
        self.equity_curve = []

    def time_call(self, func, *args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        self.prediction_latencies.append(elapsed)
        return result

    def add_prediction(self, pred: float, target: float):
        self.predictions.append(float(pred))
        self.targets.append(float(target))

    def add_equity(self, equity: float):
        self.equity_curve.append(float(equity))

    def _safe_corr(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if len(x) < 3 or len(y) < 3:
            return 0.0
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0

        corr = np.corrcoef(x, y)[0, 1]
        return float(0.0 if np.isnan(corr) else corr)

    def _rankdata(self, x):
        x = np.asarray(x)
        sorter = np.argsort(x)
        inv = np.empty(len(x), dtype=int)
        inv[sorter] = np.arange(len(x))

        x_sorted = x[sorter]
        obs = np.r_[True, x_sorted[1:] != x_sorted[:-1]]
        dense_rank = obs.cumsum() - 1

        counts = np.bincount(dense_rank)
        cumulative = np.cumsum(counts)
        starts = cumulative - counts
        avg_ranks = (starts + cumulative - 1) / 2.0 + 1.0

        return avg_ranks[dense_rank][inv]

    def compute_sharpe(self):
        if len(self.equity_curve) < 2:
            return 0.0

        equity = np.array(self.equity_curve, dtype=float)
        rets = np.diff(equity)

        if len(rets) < 2:
            return 0.0

        std = np.std(rets)
        if std == 0:
            return 0.0

        return float(np.mean(rets) / std)

    def compute_max_drawdown(self):
        if len(self.equity_curve) < 2:
            return 0.0

        equity = np.array(self.equity_curve, dtype=float)
        running_max = np.maximum.accumulate(equity)
        drawdowns = equity - running_max
        return float(drawdowns.min())

    def compute_hit_ratio(self):
        if not self.predictions or not self.targets:
            return 0.0

        preds = np.array(self.predictions, dtype=float)
        targets = np.array(self.targets, dtype=float)

        n = min(len(preds), len(targets))
        preds = preds[:n]
        targets = targets[:n]

        return float(np.mean(np.sign(preds) == np.sign(targets)))

    def compute_ic(self, base_horizon: int = 1, step: int = 1, max_horizon: int = 20):
        if len(self.predictions) < max_horizon + 5:
            return {}

        preds = np.array(self.predictions, dtype=float)
        targets = np.array(self.targets, dtype=float)

        n = min(len(preds), len(targets))
        horizons = range(base_horizon, max_horizon + 1, step)

        out = {}
        for h in horizons:
            if n - h <= 5:
                break
            p = preds[: n - h]
            t = targets[h:n]
            out[f"horizon_{h}"] = self._safe_corr(p, t)

        return out

    def compute_rank_ic(self, base_horizon: int = 1, step: int = 1, max_horizon: int = 20):
        if len(self.predictions) < max_horizon + 5:
            return {}

        preds = np.array(self.predictions, dtype=float)
        targets = np.array(self.targets, dtype=float)

        n = min(len(preds), len(targets))
        horizons = range(base_horizon, max_horizon + 1, step)

        out = {}
        for h in horizons:
            if n - h <= 5:
                break
            p = preds[: n - h]
            t = targets[h:n]
            rp = self._rankdata(p)
            rt = self._rankdata(t)
            out[f"horizon_{h}"] = self._safe_corr(rp, rt)

        return out

    def compute_rolling_ic(self, window: int = 100, horizon: int = 1):
        if len(self.predictions) < window + horizon + 5:
            return []

        preds = np.array(self.predictions, dtype=float)
        targets = np.array(self.targets, dtype=float)
        n = min(len(preds), len(targets))

        values = []
        for end in range(window + horizon, n + 1):
            p = preds[end - window - horizon : end - horizon]
            t = targets[end - window : end]
            values.append(self._safe_corr(p, t))

        return values

    def compute_ic_half_life(self, ic_dict: dict):
        if not ic_dict:
            return None

        items = []
        for k, v in ic_dict.items():
            try:
                h = int(k.split("_")[1])
                items.append((h, float(v)))
            except Exception:
                continue

        if not items:
            return None

        items.sort(key=lambda x: x[0])
        first_h, first_ic = items[0]

        if first_ic <= 0:
            return None

        threshold = first_ic / 2.0
        for h, ic in items:
            if ic <= threshold:
                return h

        return None

    def compute_signal_decay(self, base_horizon: int = 1, step: int = 1, max_horizon: int = 20):
        return self.compute_ic(
            base_horizon=base_horizon,
            step=step,
            max_horizon=max_horizon,
        )

    def summary(self):
        avg_latency = (
            sum(self.prediction_latencies) / len(self.prediction_latencies)
            if self.prediction_latencies else 0.0
        )

        mse = 0.0
        if self.predictions:
            mse = float(np.mean((np.array(self.predictions) - np.array(self.targets)) ** 2))

        sharpe = self.compute_sharpe()
        max_drawdown = self.compute_max_drawdown()
        hit_ratio = self.compute_hit_ratio()

        ic = self.compute_ic(
            base_horizon=5,
            step=5,
            max_horizon=100,
        )
        rank_ic = self.compute_rank_ic(
            base_horizon=5,
            step=5,
            max_horizon=100,
        )
        rolling_ic = self.compute_rolling_ic(
            window=100,
            horizon=5,
        )
        ic_half_life = self.compute_ic_half_life(ic)

        return {
            "avg_latency_sec": float(avg_latency),
            "mse": float(mse),
            "n_predictions": len(self.predictions),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "hit_ratio": float(hit_ratio),
            "information_coefficient": ic,
            "rank_information_coefficient": rank_ic,
            "rolling_ic_last": float(rolling_ic[-1]) if rolling_ic else 0.0,
            "rolling_ic_series": rolling_ic,
            "ic_half_life": ic_half_life,
            "signal_decay": ic,
        }