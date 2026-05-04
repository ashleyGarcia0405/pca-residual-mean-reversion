from __future__ import annotations

import pandas as pd


def align_weights_and_returns(weights: pd.DataFrame, returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    shared = weights.columns.intersection(returns.columns)
    w = weights[shared].fillna(0.0)
    r = returns[shared].reindex(w.index)
    return w, r


def backtest(weights: pd.DataFrame, returns: pd.DataFrame, cost_bps: float = 0.0) -> pd.DataFrame:
    c = cost_bps * 1e-4
    w, r = align_weights_and_returns(weights, returns)

    r_next = r.shift(-1)
    gross = (w * r_next).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1)
    net = gross - c * turnover

    return pd.DataFrame({"gross": gross, "net": net, "turnover": turnover}).dropna()


def backtest_holding_period(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    holding_days: int = 5,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    if holding_days <= 0:
        raise ValueError("holding_days must be positive")

    c = cost_bps * 1e-4
    w, r = align_weights_and_returns(weights, returns)

    fwd_ret = r.shift(-1).rolling(holding_days).sum().shift(-(holding_days - 1))
    gross = (w * fwd_ret).sum(axis=1) / holding_days
    turnover = w.diff().abs().sum(axis=1)
    net = gross - c * turnover

    return pd.DataFrame({"gross": gross, "net": net, "turnover": turnover}).dropna()
