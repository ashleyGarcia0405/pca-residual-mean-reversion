from __future__ import annotations

import numpy as np
import pandas as pd


def _renormalize_gross(weights: pd.DataFrame | pd.Series, gross_target: float) -> pd.DataFrame | pd.Series:
    if isinstance(weights, pd.DataFrame):
        gross = weights.abs().sum(axis=1).replace(0, np.nan)
        return weights.div(gross, axis=0) * gross_target

    gross = weights.abs().sum()
    if gross == 0 or not np.isfinite(gross):
        return weights * np.nan
    return weights / gross * gross_target


def build_weights_equal(
    zscores: pd.DataFrame,
    c: float = 1.0,
    G: float = 1.0,
    w_max: float | None = None,
    invert: bool = False,
) -> pd.DataFrame:
    long_mask = (zscores < -c).astype(float)
    short_mask = (zscores > c).astype(float)

    weights = long_mask.div(long_mask.sum(axis=1).clip(lower=1), axis=0) * (G / 2)
    weights = weights - short_mask.div(short_mask.sum(axis=1).clip(lower=1), axis=0) * (G / 2)

    if invert:
        weights = -weights

    if w_max is not None:
        weights = weights.clip(-w_max, w_max)
        weights = _renormalize_gross(weights, G)

    return weights.fillna(0.0)


def build_weights_signal(
    zscores: pd.DataFrame,
    c: float = 1.0,
    G: float = 1.0,
    w_max: float | None = None,
    invert: bool = False,
) -> pd.DataFrame:
    signal = -zscores
    if invert:
        signal = -signal

    long_mask = (signal > c).astype(float)
    short_mask = (signal < -c).astype(float)

    long_strength = (signal * long_mask).clip(lower=0)
    short_strength = (-signal * short_mask).clip(lower=0)

    weights = long_strength.div(long_strength.sum(axis=1).clip(lower=1e-12), axis=0) * (G / 2)
    weights = weights - short_strength.div(short_strength.sum(axis=1).clip(lower=1e-12), axis=0) * (G / 2)

    if w_max is not None:
        weights = weights.clip(-w_max, w_max)
        weights = _renormalize_gross(weights, G)

    return weights.fillna(0.0)


def rebalance_every_n_days(weights: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    if n <= 0:
        raise ValueError("n must be positive")
    rebalanced = weights.copy() * np.nan
    rebalanced.iloc[::n] = weights.iloc[::n]
    return rebalanced.ffill().fillna(0.0)


def apply_no_trade_band(target_weights: pd.DataFrame, band: float = 0.0025) -> pd.DataFrame:
    actual = target_weights.copy()
    if actual.empty:
        return actual

    actual.iloc[0] = target_weights.iloc[0].fillna(0.0)
    for t in range(1, len(target_weights)):
        prev = actual.iloc[t - 1]
        target = target_weights.iloc[t].fillna(0.0)
        change = (target - prev).abs()
        actual.iloc[t] = prev.where(change < band, target)

    return actual.fillna(0.0)


def smooth_weights(weights: pd.DataFrame, alpha: float = 0.2, gross_target: float = 1.0) -> pd.DataFrame:
    smoothed = weights.ewm(alpha=alpha).mean()
    smoothed = _renormalize_gross(smoothed, gross_target)
    return smoothed.fillna(0.0)


def build_cross_sectional_reversal(
    log_returns: pd.DataFrame,
    m: int = 5,
    c: float = 1.0,
    G: float = 1.0,
    w_max: float | None = None,
) -> pd.DataFrame:
    signal = -log_returns.rolling(m).sum().shift(1)
    signal = signal.sub(signal.mean(axis=1), axis=0)
    signal = signal.div(signal.std(axis=1).replace(0, np.nan), axis=0)
    return build_weights_equal(signal, c=c, G=G, w_max=w_max)


def build_weights_quantile(
    zscores: pd.DataFrame,
    q: float = 0.10,
    G: float = 1.0,
    invert: bool = False,
) -> pd.DataFrame:
    sig = -zscores if not invert else zscores
    n = sig.notna().sum(axis=1)
    k_each = (n * q).clip(lower=1).astype(int)
    rank_asc = sig.rank(axis=1, method="first")
    rank_desc = sig.rank(axis=1, method="first", ascending=False)
    long_mask = rank_asc.le(k_each, axis=0).astype(float)
    short_mask = rank_desc.le(k_each, axis=0).astype(float)
    w = long_mask.div(long_mask.sum(axis=1).clip(lower=1), axis=0) * (G / 2)
    w = w - short_mask.div(short_mask.sum(axis=1).clip(lower=1), axis=0) * (G / 2)
    return w.fillna(0.0)


def build_weights_sparse_composite(
    zscores: pd.DataFrame,
    k_rev: int = 10,
    k_cont: int = 3,
    G: float = 1.0,
) -> pd.DataFrame:
    return build_weights_asymmetric_composite(
        zscores,
        k_rev_long=k_rev,
        k_rev_short=k_rev,
        k_cont_long=k_cont,
        k_cont_short=k_cont,
        G=G,
    )


def build_weights_asymmetric_composite(
    zscores: pd.DataFrame,
    k_rev_long: int = 6,
    k_rev_short: int = 14,
    k_cont_long: int = 2,
    k_cont_short: int = 2,
    G: float = 1.0,
) -> pd.DataFrame:
    rank_lo = zscores.rank(axis=1, method="first")
    rank_hi = zscores.rank(axis=1, method="first", ascending=False)
    rev_long = rank_lo.le(k_rev_long, axis=0).astype(float)
    rev_short = rank_hi.le(k_rev_short, axis=0).astype(float)
    cont_long = rank_hi.le(k_cont_long, axis=0).astype(float)
    cont_short = rank_lo.le(k_cont_short, axis=0).astype(float)
    long_combined = (rev_long - cont_short).clip(lower=0) + cont_long
    short_combined = (rev_short - cont_long).clip(lower=0) + cont_short
    w = long_combined.div(long_combined.sum(axis=1).clip(lower=1), axis=0) * (G / 2)
    w = w - short_combined.div(short_combined.sum(axis=1).clip(lower=1), axis=0) * (G / 2)
    return w.fillna(0.0)


def neutralize_single_day_weights_safe(
    w_row: pd.Series,
    B_t: pd.DataFrame,
    G: float = 1.0,
    w_max: float | None = None,
    cond_max: float = 1e8,
) -> pd.Series:
    tickers = w_row.index.intersection(B_t.index)
    w = w_row.loc[tickers].astype(float)
    B = B_t.loc[tickers].astype(float)
    valid = np.isfinite(w.values) & np.isfinite(B.values).all(axis=1)
    out = pd.Series(0.0, index=w_row.index, dtype=float)

    if valid.sum() <= B.shape[1]:
        return out

    tickers_valid = w.index[valid]
    wv = w.loc[tickers_valid].values
    Bv = B.loc[tickers_valid].values

    try:
        if not np.isfinite(Bv).all() or not np.isfinite(wv).all():
            return out
        if np.linalg.cond(Bv) > cond_max:
            return out
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            beta = np.linalg.lstsq(Bv, wv, rcond=None)[0]
            w_neut = wv - Bv @ beta
    except Exception:
        return out

    if not np.isfinite(w_neut).all():
        return out

    w_neut = w_neut - w_neut.mean()
    if w_max is not None:
        w_neut = np.clip(w_neut, -w_max, w_max)

    gross = np.abs(w_neut).sum()
    if gross > 0 and np.isfinite(gross):
        w_neut = w_neut / gross * G
        out.loc[tickers_valid] = w_neut

    return out


def neutralize_weights_over_time(
    weights: pd.DataFrame,
    rolling_loadings: dict,
    G: float = 1.0,
    w_max: float | None = None,
    cond_max: float = 1e8,
) -> pd.DataFrame:
    rows = []
    for dt, w_row in weights.iterrows():
        if dt not in rolling_loadings:
            rows.append(pd.Series(0.0, index=weights.columns, name=dt, dtype=float))
            continue
        w_neut = neutralize_single_day_weights_safe(
            w_row,
            rolling_loadings[dt],
            G=G,
            w_max=w_max,
            cond_max=cond_max,
        )
        w_neut.name = dt
        rows.append(w_neut)
    return pd.DataFrame(rows, index=weights.index, columns=weights.columns).fillna(0.0)


def compute_factor_exposures(weights: pd.DataFrame, rolling_loadings: dict) -> pd.DataFrame:
    rows = []
    for dt, w_row in weights.iterrows():
        if dt not in rolling_loadings:
            continue
        B_t = rolling_loadings[dt]
        tickers = w_row.index.intersection(B_t.index)
        exp = B_t.loc[tickers].mul(w_row.loc[tickers], axis=0).sum(axis=0)
        exp.name = dt
        rows.append(exp)
    return pd.DataFrame(rows) if rows else pd.DataFrame()
