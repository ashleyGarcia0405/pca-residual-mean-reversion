"""Run a single experiment from a JSON spec and append one row to the ledger.

Usage:
    .venv/bin/python run_experiment.py --spec-json '{"id":"e000",...}'
    .venv/bin/python run_experiment.py --spec-file path/to/spec.json
    .venv/bin/python run_experiment.py --queue-line 0           # read line N from backlog.jsonl

Spec schema (all keys optional unless noted):
    id          required, e.g. "e007"
    phase       required, "A".."G"
    family      required, e.g. "holding_period", "threshold", "smoothing", "timing", "pca_grid", "quantile", "universe"
    pca_suffix  required, e.g. "_L126_K10_M20"
    weight_type "equal" (default) | "signal" | "quantile" | "factor_neutral_equal"
    invert      bool (default true)
    c           threshold (default 1.0)
    q           quantile (default 0.10) — only for weight_type=quantile
    w_max       max single-name weight (default 0.05)
    G           gross exposure (default 1.0)
    smooth_alpha   None to disable (default 0.3)
    band           None to disable (default 0.0025)
    holding_days   None for daily (default None)
    notes       free text
"""
from __future__ import annotations
import argparse, hashlib, json, os, pickle, sys, time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pca_residual_mean_reversion.backtest import backtest, backtest_holding_period
from pca_residual_mean_reversion.metrics import perf_metrics
from pca_residual_mean_reversion.portfolio import (
    build_weights_equal,
    build_weights_signal,
    smooth_weights,
    apply_no_trade_band,
    neutralize_weights_over_time,
    rebalance_every_n_days,
)

PROCESSED = ROOT / "data" / "processed"
EXP_DIR = ROOT / "experiments"
LEDGER = EXP_DIR / "ledger.jsonl"
LOCK_DIR = EXP_DIR / "locks"

DOWNSTREAM_KEYS = ["weight_type", "invert", "c", "q", "w_max", "G",
                   "smooth_alpha", "band", "holding_days", "reb_n_days"]


def downstream_hash(spec: dict) -> str:
    payload = {k: spec.get(k) for k in DOWNSTREAM_KEYS}
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:8]


def acquire_lock(lock_path: Path) -> bool:
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def append_ledger(row: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as fh:
        fh.write(json.dumps(row) + "\n")


def build_quantile_weights(zscores: pd.DataFrame, q: float, G: float, invert: bool) -> pd.DataFrame:
    """Top/bottom q quantile each day, equal-weighted, dollar-neutral."""
    sig = -zscores if not invert else zscores
    n = sig.notna().sum(axis=1)
    k_long = (n * q).clip(lower=1).astype(int)

    rank_asc = sig.rank(axis=1, method="first")
    rank_desc = sig.rank(axis=1, method="first", ascending=False)
    long_mask = rank_asc.le(k_long, axis=0).astype(float)
    short_mask = rank_desc.le(k_long, axis=0).astype(float)
    w = long_mask.div(long_mask.sum(axis=1).clip(lower=1), axis=0) * (G / 2)
    w = w - short_mask.div(short_mask.sum(axis=1).clip(lower=1), axis=0) * (G / 2)
    return w.fillna(0.0)


def run(spec: dict) -> dict:
    started = time.time()
    sp = {
        "weight_type": "equal",
        "invert": True,
        "c": 1.0,
        "q": 0.10,
        "w_max": 0.05,
        "G": 1.0,
        "smooth_alpha": 0.3,
        "band": 0.0025,
        "holding_days": None,
        "reb_n_days": None,
        **spec,
    }

    suffix = sp["pca_suffix"]
    dh = downstream_hash(sp)
    lock = LOCK_DIR / f"{suffix}.{dh}.lock"

    base = {
        "id": sp["id"],
        "ts": datetime.now(timezone.utc).isoformat(),
        "phase": sp.get("phase"),
        "family": sp.get("family"),
        "pca_suffix": suffix,
        "params": {k: sp[k] for k in DOWNSTREAM_KEYS},
        "downstream_hash": dh,
        "notes": sp.get("notes", ""),
    }

    if not acquire_lock(lock):
        row = {**base, "status": "duplicate", "elapsed_s": round(time.time() - started, 2)}
        append_ledger(row)
        return row

    zs_path = PROCESSED / f"residual_zscores{suffix}.parquet"
    lr_path = PROCESSED / "log_returns.parquet"
    if not zs_path.exists():
        row = {**base, "status": "missing_pca", "elapsed_s": round(time.time() - started, 2),
               "notes": f"missing {zs_path.name}; run residuals generate-spec first"}
        append_ledger(row)
        return row

    zscores = pd.read_parquet(zs_path)
    log_returns = pd.read_parquet(lr_path)
    shared = zscores.columns.intersection(log_returns.columns)
    zscores = zscores[shared]
    log_returns = log_returns[shared]

    wt = sp["weight_type"]
    if wt == "equal":
        w = build_weights_equal(zscores, c=sp["c"], G=sp["G"], w_max=sp["w_max"], invert=sp["invert"])
    elif wt == "signal":
        w = build_weights_signal(zscores, c=sp["c"], G=sp["G"], w_max=sp["w_max"], invert=sp["invert"])
    elif wt == "quantile":
        w = build_quantile_weights(zscores, q=sp["q"], G=sp["G"], invert=sp["invert"])
    elif wt == "factor_neutral_equal":
        loadings_pkl = PROCESSED / f"pca_rolling_loadings{suffix}.pkl"
        with open(loadings_pkl, "rb") as f:
            rolling_loadings = pickle.load(f)["loadings_by_date"]
        w_raw = build_weights_equal(zscores, c=sp["c"], G=sp["G"], w_max=sp["w_max"], invert=sp["invert"])
        w = neutralize_weights_over_time(w_raw, rolling_loadings, G=sp["G"], w_max=sp["w_max"])
    else:
        row = {**base, "status": "bad_weight_type", "notes": wt}
        append_ledger(row)
        return row

    if sp["smooth_alpha"] is not None:
        w = smooth_weights(w, alpha=sp["smooth_alpha"], gross_target=sp["G"])
    if sp["band"] is not None and sp["band"] > 0:
        w = apply_no_trade_band(w, band=sp["band"])
    if sp["reb_n_days"] is not None and sp["reb_n_days"] > 1:
        w = rebalance_every_n_days(w, n=int(sp["reb_n_days"]))

    if sp["holding_days"] is not None:
        bt0 = backtest_holding_period(w, log_returns, holding_days=sp["holding_days"], cost_bps=0)
        bt1 = backtest_holding_period(w, log_returns, holding_days=sp["holding_days"], cost_bps=1)
        bt3 = backtest_holding_period(w, log_returns, holding_days=sp["holding_days"], cost_bps=3)
        bt5 = backtest_holding_period(w, log_returns, holding_days=sp["holding_days"], cost_bps=5)
    else:
        bt0 = backtest(w, log_returns, cost_bps=0)
        bt1 = backtest(w, log_returns, cost_bps=1)
        bt3 = backtest(w, log_returns, cost_bps=3)
        bt5 = backtest(w, log_returns, cost_bps=5)

    pm0 = perf_metrics(bt0["gross"])
    pm1 = perf_metrics(bt1["net"])
    pm3 = perf_metrics(bt3["net"])
    pm5 = perf_metrics(bt5["net"])

    row = {
        **base,
        "status": "ok",
        "sharpe_gross": round(float(pm0["Sharpe"]), 3),
        "sharpe_net_1bps": round(float(pm1["Sharpe"]), 3),
        "sharpe_net_3bps": round(float(pm3["Sharpe"]), 3),
        "sharpe_net_5bps": round(float(pm5["Sharpe"]), 3),
        "ann_ret_gross": round(float(pm0["Ann. Return"]), 4),
        "ann_ret_net_3bps": round(float(pm3["Ann. Return"]), 4),
        "ann_ret_net_5bps": round(float(pm5["Ann. Return"]), 4),
        "turnover": round(float(bt0["turnover"].mean()), 4),
        "max_dd_net_5bps": round(float(pm5["Max Drawdown"]), 4),
        "hit_rate_net_5bps": round(float(pm5["Hit Rate"]), 4),
        "n_days": int(pm0["N Days"]),
        "elapsed_s": round(time.time() - started, 2),
    }
    append_ledger(row)
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--spec-json", type=str)
    g.add_argument("--spec-file", type=str)
    g.add_argument("--queue-line", type=int)
    args = p.parse_args()

    if args.spec_json:
        spec = json.loads(args.spec_json)
    elif args.spec_file:
        spec = json.loads(Path(args.spec_file).read_text())
    else:
        with open(EXP_DIR / "backlog.jsonl") as fh:
            for i, line in enumerate(fh):
                if i == args.queue_line:
                    spec = json.loads(line)
                    break

    row = run(spec)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
