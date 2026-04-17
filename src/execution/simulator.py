class ExecutionSimulator:
    def __init__(self, initial_capital: float, transaction_cost: float, slippage: float, position_size: int):
        self.cash = initial_capital
        self.position = 0
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.position_size = position_size
        self.last_mid = None
        self.realized_pnl = 0.0
        self.trade_count = 0

    def mark_to_market(self, mid: float):
        self.last_mid = mid
        return self.cash + self.position * mid

    def trade_to_target(self, target_position: int, bid: float, ask: float):
        if target_position == self.position:
            return

        delta = target_position - self.position

        if delta > 0:
            fill_price = ask + self.slippage
            self.cash -= delta * fill_price
        else:
            fill_price = bid - self.slippage
            self.cash += abs(delta) * fill_price

        self.cash -= self.transaction_cost * abs(delta)
        self.position = target_position
        self.trade_count += abs(delta)

    def snapshot(self, mid: float):
        equity = self.cash + self.position * mid
        return {
            "cash": float(self.cash),
            "position": int(self.position),
            "mid": float(mid),
            "equity": float(equity),
            "trade_count": int(self.trade_count),
        }