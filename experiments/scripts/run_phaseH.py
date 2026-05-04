"""Phase H: sparse selection, composite signals, z-score smoothing.

Runs ~25 variants on the timing-fixed PCA artifacts and appends to the ledger.
"""
from __future__ import annotations
import json, sys, hashlib, os, time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pca_residual_mean_reversion.backtest import backtest
from pca_residual_mean_reversion.metrics import perf_metrics
from pca_residual_mean_reversion.portfolio import smooth_weights, apply_no_trade_band

PROCESSED = ROOT / "data" / "processed"
LEDGER = ROOT / "experiments" / "ledger.jsonl"

SUF = "_L126_K10_M20_t0"  # timing-fixed PCA
zscores = pd.read_parquet(PROCESSED / f"residual_zscores{SUF}.parquet")
log_returns = pd.read_parquet(PROCESSED / "log_returns.parquet")
shared = zscores.columns.intersection(log_returns.columns)
zscores = zscores[shared]
log_returns = log_returns[shared]


def _renorm(w, gross=1.0):
    g = w.abs().sum(axis=1).replace(0, np.nan)
    return w.div(g, axis=0) * gross


def sparse_weights(zs, k_each=20, invert=False, gross=1.0):
    """Trade exactly the top-k longs and top-k shorts each day by signal strength."""
    sig = -zs if not invert else zs
    rank_long = sig.rank(axis=1, method="first")          # smallest sig (most oversold) = rank 1
    rank_short = sig.rank(axis=1, method="first", ascending=False)
    long_mask = rank_long.le(k_each, axis=0).astype(float)
    short_mask = rank_short.le(k_each, axis=0).astype(float)
    w = long_mask.div(long_mask.sum(axis=1).clip(lower=1), axis=0) * (gross / 2)
    w = w - short_mask.div(short_mask.sum(axis=1).clip(lower=1), axis=0) * (gross / 2)
    return w.fillna(0.0)


def threshold_weights(zs, c=1.0, w_max=0.05, invert=False, gross=1.0):
    long_mask = (zs < -c).astype(float) if not invert else (zs > c).astype(float)
    short_mask = (zs > c).astype(float) if not invert else (zs < -c).astype(float)
    w = long_mask.div(long_mask.sum(axis=1).clip(lower=1), axis=0) * (gross / 2)
    w = w - short_mask.div(short_mask.sum(axis=1).clip(lower=1), axis=0) * (gross / 2)
    if w_max:
        w = w.clip(-w_max, w_max)
        w = _renorm(w, gross)
    return w.fillna(0.0)


def composite_weights(zs, c_lo=1.0, c_hi=2.5, w_max=0.05, gross=1.0,
                      reversion_weight=0.5, continuation_weight=0.5):
    """Combine moderate-z mean reversion (c_lo<|z|<c_hi, no-invert) with
    extreme-z continuation (|z|>c_hi, invert)."""
    rev = pd.DataFrame(0.0, index=zs.index, columns=zs.columns)
    cont = pd.DataFrame(0.0, index=zs.index, columns=zs.columns)
    # mean-reversion side: trade names where c_lo<|z|<c_hi, long oversold (z<-c_lo)
    rev_long  = ((zs < -c_lo) & (zs > -c_hi)).astype(float)
    rev_short = ((zs >  c_lo) & (zs <  c_hi)).astype(float)
    rev = rev_long.div(rev_long.sum(axis=1).clip(lower=1), axis=0) * 0.5
    rev = rev - rev_short.div(rev_short.sum(axis=1).clip(lower=1), axis=0) * 0.5
    # continuation side: trade names where |z|>c_hi, long overbought (z>c_hi)
    cont_long  = (zs >  c_hi).astype(float)
    cont_short = (zs < -c_hi).astype(float)
    cont = cont_long.div(cont_long.sum(axis=1).clip(lower=1), axis=0) * 0.5
    cont = cont - cont_short.div(cont_short.sum(axis=1).clip(lower=1), axis=0) * 0.5
    # blend
    w = rev * reversion_weight + cont * continuation_weight
    if w_max:
        w = w.clip(-w_max, w_max)
    w = _renorm(w, gross).fillna(0.0)
    return w


def smooth_z(zs, alpha):
    return zs.ewm(alpha=alpha).mean()


def append_ledger(row):
    with open(LEDGER, "a") as fh:
        fh.write(json.dumps(row) + "\n")


def evaluate(spec_id, family, notes, weights, params):
    bt0 = backtest(weights, log_returns, cost_bps=0)
    bt1 = backtest(weights, log_returns, cost_bps=1)
    bt3 = backtest(weights, log_returns, cost_bps=3)
    bt5 = backtest(weights, log_returns, cost_bps=5)
    pm0, pm1, pm3, pm5 = (perf_metrics(bt0["gross"]), perf_metrics(bt1["net"]),
                          perf_metrics(bt3["net"]), perf_metrics(bt5["net"]))
    row = {
        "id": spec_id, "ts": datetime.now(timezone.utc).isoformat(),
        "phase": "H", "family": family,
        "pca_suffix": SUF, "params": params, "notes": notes, "status": "ok",
        "sharpe_gross":   round(float(pm0["Sharpe"]), 3),
        "sharpe_net_1bps":round(float(pm1["Sharpe"]), 3),
        "sharpe_net_3bps":round(float(pm3["Sharpe"]), 3),
        "sharpe_net_5bps":round(float(pm5["Sharpe"]), 3),
        "ann_ret_gross":  round(float(pm0["Ann. Return"]), 4),
        "ann_ret_net_3bps": round(float(pm3["Ann. Return"]), 4),
        "ann_ret_net_5bps": round(float(pm5["Ann. Return"]), 4),
        "turnover":      round(float(bt0["turnover"].mean()), 4),
        "max_dd_net_5bps":round(float(pm5["Max Drawdown"]), 4),
        "hit_rate_net_5bps":round(float(pm5["Hit Rate"]), 4),
        "n_days": int(pm0["N Days"]),
    }
    append_ledger(row)
    return row


# ────────────────── Sparse selection (mean reversion, no-invert) ──────────────────
print("Sparse mean-reversion variants (timing-fix, no-invert):")
for k in [10, 20, 30, 50]:
    w = sparse_weights(zscores, k_each=k, invert=False)
    r = evaluate(f"h_sp{k}", "sparse", f"sparse top-{k} L/S, no smooth",
                 w, {"k_each": k, "invert": False})
    print(f"  h_sp{k:<3}  gross={r['sharpe_gross']:+.2f}  n3={r['sharpe_net_3bps']:+.2f}  n5={r['sharpe_net_5bps']:+.2f}  turn={r['turnover']:.2f}")

print("\nSparse + smoothing (no-invert):")
for k, alpha in [(20, 0.3), (20, 0.15), (20, 0.05), (10, 0.3), (10, 0.15), (10, 0.05)]:
    w = sparse_weights(zscores, k_each=k, invert=False)
    w = smooth_weights(w, alpha=alpha, gross_target=1.0)
    r = evaluate(f"h_sp{k}_a{int(alpha*100)}", "sparse",
                 f"sparse top-{k} + EWMA({alpha})",
                 w, {"k_each": k, "invert": False, "smooth_alpha": alpha})
    print(f"  k={k} a={alpha:.2f}  gross={r['sharpe_gross']:+.2f}  n3={r['sharpe_net_3bps']:+.2f}  n5={r['sharpe_net_5bps']:+.2f}  turn={r['turnover']:.2f}")

print("\nSparse + invert (extreme-z continuation):")
for k in [10, 20, 30]:
    w = sparse_weights(zscores, k_each=k, invert=True)
    w = smooth_weights(w, alpha=0.05, gross_target=1.0)
    w = apply_no_trade_band(w, band=0.005)
    r = evaluate(f"h_spi{k}", "sparse_invert",
                 f"sparse top-{k} invert + smooth",
                 w, {"k_each": k, "invert": True, "smooth_alpha": 0.05, "band": 0.005})
    print(f"  spi{k:<2}  gross={r['sharpe_gross']:+.2f}  n3={r['sharpe_net_3bps']:+.2f}  n5={r['sharpe_net_5bps']:+.2f}  turn={r['turnover']:.2f}")

# ────────────────── Composite signal ──────────────────
print("\nComposite (moderate-z reversion + extreme-z continuation):")
for c_lo, c_hi, rw, cw in [
    (1.0, 2.5, 0.5, 0.5),
    (1.0, 2.5, 0.7, 0.3),
    (1.0, 2.5, 0.3, 0.7),
    (0.75, 2.5, 0.5, 0.5),
    (1.0, 2.0, 0.5, 0.5),
    (1.5, 3.0, 0.5, 0.5),
    (1.0, 2.5, 1.0, 0.0),  # pure reversion baseline
    (1.0, 2.5, 0.0, 1.0),  # pure continuation baseline
]:
    w = composite_weights(zscores, c_lo=c_lo, c_hi=c_hi,
                          reversion_weight=rw, continuation_weight=cw)
    w = smooth_weights(w, alpha=0.15, gross_target=1.0)
    w = apply_no_trade_band(w, band=0.005)
    r = evaluate(f"h_comp_{c_lo}_{c_hi}_{int(rw*10)}{int(cw*10)}", "composite",
                 f"comp lo={c_lo} hi={c_hi} rev={rw:.1f} cont={cw:.1f}",
                 w, {"c_lo": c_lo, "c_hi": c_hi, "rev_w": rw, "cont_w": cw,
                     "smooth_alpha": 0.15, "band": 0.005})
    print(f"  lo={c_lo} hi={c_hi} rev={rw:.1f}/cont={cw:.1f}  gross={r['sharpe_gross']:+.2f}  n3={r['sharpe_net_3bps']:+.2f}  n5={r['sharpe_net_5bps']:+.2f}  turn={r['turnover']:.2f}")

# ────────────────── z-score smoothing ──────────────────
print("\nz-score smoothing (no-invert, c=1.0):")
for alpha in [0.5, 0.3, 0.15]:
    zs_s = smooth_z(zscores, alpha=alpha)
    w = threshold_weights(zs_s, c=1.0, invert=False)
    w = smooth_weights(w, alpha=0.15, gross_target=1.0)
    w = apply_no_trade_band(w, band=0.005)
    r = evaluate(f"h_zs_{int(alpha*100)}", "z_smooth",
                 f"z-smooth alpha={alpha} + thresh c=1.0",
                 w, {"smooth_z_alpha": alpha, "c": 1.0, "invert": False})
    print(f"  z-alpha={alpha}  gross={r['sharpe_gross']:+.2f}  n3={r['sharpe_net_3bps']:+.2f}  n5={r['sharpe_net_5bps']:+.2f}  turn={r['turnover']:.2f}")

# ────────────────── Combine sparse + composite ──────────────────
print("\nSparse composite (top-K mean-revert + top-K' extreme-cont):")
def sparse_composite(zs, k_rev=15, k_cont=8, gross=1.0):
    """Top-k_rev moderate-z names (mean revert) + top-k_cont extreme names (continuation)."""
    # Mean revert: pick top-k_rev by smallest z (most oversold) and largest z (most overbought)
    # Continuation: pick top-k_cont by *most extreme* z (sign matching continuation direction)
    rank_lo = zs.rank(axis=1, method='first')  # rank 1 = most negative (oversold)
    rank_hi = zs.rank(axis=1, method='first', ascending=False)
    # Reversion: long bottom k_rev (oversold), short top k_rev (overbought)
    rev_long = rank_lo.le(k_rev, axis=0).astype(float)
    rev_short = rank_hi.le(k_rev, axis=0).astype(float)
    # Continuation: long top k_cont (overbought), short bottom k_cont (oversold)
    cont_long = rank_hi.le(k_cont, axis=0).astype(float)
    cont_short = rank_lo.le(k_cont, axis=0).astype(float)
    # Net (cont overrides rev when same name)
    long_combined = (rev_long - cont_short).clip(lower=0)  # name in rev_long & not cont_short
    short_combined = (rev_short - cont_long).clip(lower=0)
    long_combined = long_combined + cont_long
    short_combined = short_combined + cont_short
    w = long_combined.div(long_combined.sum(axis=1).clip(lower=1), axis=0) * (gross/2)
    w = w - short_combined.div(short_combined.sum(axis=1).clip(lower=1), axis=0) * (gross/2)
    return w.fillna(0.0)

for k_rev, k_cont in [(15, 8), (20, 10), (10, 5), (25, 5), (15, 5)]:
    w = sparse_composite(zscores, k_rev=k_rev, k_cont=k_cont)
    w = smooth_weights(w, alpha=0.05, gross_target=1.0)
    w = apply_no_trade_band(w, band=0.005)
    r = evaluate(f"h_sc_{k_rev}_{k_cont}", "sparse_composite",
                 f"sparse-comp rev top-{k_rev} cont top-{k_cont}",
                 w, {"k_rev": k_rev, "k_cont": k_cont, "smooth_alpha": 0.05, "band": 0.005})
    print(f"  rev={k_rev:<2} cont={k_cont:<2}  gross={r['sharpe_gross']:+.2f}  n3={r['sharpe_net_3bps']:+.2f}  n5={r['sharpe_net_5bps']:+.2f}  turn={r['turnover']:.2f}")

print("\nDone.")
