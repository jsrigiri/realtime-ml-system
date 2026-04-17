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


def plot_predictions_vs_realized(
    predictions,
    realized,
    output_path: str = "artifacts/pred_vs_realized.png",
):
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


def plot_batch_session_pnl(
    session_ids,
    session_pnls,
    output_path: str = "artifacts/batch_session_pnl.png",
):
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