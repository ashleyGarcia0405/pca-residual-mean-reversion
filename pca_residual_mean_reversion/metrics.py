from __future__ import annotations

import numpy as np
import pandas as pd


def perf_metrics(returns: pd.Series, ann: int = 252) -> dict:
    r = returns.dropna()
    if r.empty:
        return {
            "Ann. Return": np.nan,
            "Ann. Vol": np.nan,
            "Sharpe": np.nan,
            "Max Drawdown": np.nan,
            "Calmar": np.nan,
            "Hit Rate": np.nan,
            "N Days": 0,
        }

    total_ret = (1 + r).prod() - 1
    ann_ret = (1 + total_ret) ** (ann / len(r)) - 1
    ann_vol = r.std() * np.sqrt(ann)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cumret = (1 + r).cumprod()
    max_dd = (cumret / cumret.cummax() - 1).min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    return {
        "Ann. Return": ann_ret,
        "Ann. Vol": ann_vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Calmar": calmar,
        "Hit Rate": (r > 0).mean(),
        "N Days": len(r),
    }
