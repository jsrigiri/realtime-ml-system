import matplotlib.pyplot as plt


def plot_equity_curve(equity_series):
    plt.figure()
    plt.plot(equity_series)
    plt.title("Equity Curve")
    plt.xlabel("Tick")
    plt.ylabel("Equity")
    plt.grid()
    plt.savefig("artifacts/equity_curve.png")
    plt.close()


def plot_predictions_vs_actual(preds, actuals):
    plt.figure()
    plt.plot(preds, label="Predictions")
    plt.plot(actuals, label="Actual")
    plt.legend()
    plt.title("Pred vs Actual")
    plt.grid()
    plt.savefig("artifacts/pred_vs_actual.png")
    plt.close()