from src.execution.simulator import ExecutionSimulator


def test_execution_trade_and_snapshot():
    sim = ExecutionSimulator(
        initial_capital=1000.0,
        transaction_cost=0.01,
        slippage=0.0,
        position_size=1,
    )

    sim.trade_to_target(target_position=1, bid=99.9, ask=100.1)
    snap = sim.snapshot(100.0)

    assert "equity" in snap
    assert snap["position"] == 1