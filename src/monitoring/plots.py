from pathlib import Path
import matplotlib.pyplot as plt


def ensure_artifacts_dir(path: str = "artifacts"):
    Path(path).mkdir(exist_ok=True)


def plot_equity_curve(equity_series, output_path: str = "artifacts/equity_curve.png"):
    if not equity_series:
        return

    ensure_artifacts_dir()
    plt.figure(figsize=(10, 5))
    plt.plot(equity_series)
    plt.title("Equity Curve")
    plt.xlabel("Processed Tick")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_predictions_vs_realized(predictions, realized, output_path: str = "artifacts/pred_vs_realized.png"):
    if not predictions or not realized:
        return

    n = min(len(predictions), len(realized))
    if n == 0:
        return

    ensure_artifacts_dir()
    plt.figure(figsize=(10, 5))
    plt.plot(predictions[:n], label="Predictions")
    plt.plot(realized[:n], label="Realized Targets")
    plt.title("Predictions vs Realized Targets")
    plt.xlabel("Prediction Index")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_batch_session_pnl(session_ids, session_pnls, output_path: str = "artifacts/batch_session_pnl.png"):
    if not session_ids or not session_pnls:
        return

    ensure_artifacts_dir()
    plt.figure(figsize=(10, 5))
    plt.plot(session_ids, session_pnls, marker="o")
    plt.title("Batch Session PnL")
    plt.xlabel("Session")
    plt.ylabel("Session PnL")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_signal_decay(signal_decay: dict, output_path: str = "artifacts/signal_decay.png"):
    if not signal_decay:
        return

    ensure_artifacts_dir()

    horizons = list(signal_decay.keys())
    values = list(signal_decay.values())

    plt.figure(figsize=(10, 5))
    plt.plot(horizons, values, marker="o")
    plt.title("Signal Decay")
    plt.xlabel("Forward Horizon (ticks)")
    plt.ylabel("Correlation (IC)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_rolling_ic(rolling_ic, output_path="artifacts/rolling_ic.png"):
    if not rolling_ic:
        return
    ensure_artifacts_dir()
    plt.figure(figsize=(10, 5))
    plt.plot(rolling_ic)
    plt.title("Rolling Information Coefficient")
    plt.xlabel("Window Index")
    plt.ylabel("IC")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_ic_curve(ic_dict, output_path="artifacts/ic_curve.png", title="Information Coefficient"):
    if not ic_dict:
        return
    ensure_artifacts_dir()
    horizons = list(ic_dict.keys())
    values = list(ic_dict.values())
    plt.figure(figsize=(10, 5))
    plt.plot(horizons, values, marker="o")
    plt.title(title)
    plt.xlabel("Forward Horizon (ticks)")
    plt.ylabel("Correlation")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()