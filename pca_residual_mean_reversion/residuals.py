from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.decomposition import PCA
from tqdm import tqdm

from pca_residual_mean_reversion.config import PROCESSED_DATA_DIR

app = typer.Typer(add_completion=False)


def output_suffix(L: int, K: int, M: int) -> str:
    return f"_L{L}_K{K}_M{M}"


def rolling_pca_residuals(returns: pd.DataFrame, L: int, k: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray | None, dict]:
    T, N = returns.shape
    dates = returns.index[L:]
    tickers = returns.columns

    resid_rows = np.full((T - L, N), np.nan)
    evr_rows = np.full((T - L, k), np.nan)
    factor_rows = np.full((T - L, k), np.nan)
    loadings_last = None
    rolling_loadings: dict = {}

    R_vals = returns.values
    pc_cols = [f"PC{j + 1}" for j in range(k)]

    for i, t in enumerate(tqdm(range(L, T), desc=f"L={L}, k={k}")):
        window = R_vals[t - L : t].copy()

        mu = window.mean(axis=0)
        sig = window.std(axis=0, ddof=1)
        sig = np.where(sig == 0, 1.0, sig)
        R_std = (window - mu) / sig

        pca = PCA(n_components=k)
        F = pca.fit_transform(R_std)

        F_aug = np.column_stack([np.ones(L), F])
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            coef, _, _, _ = np.linalg.lstsq(F_aug, window, rcond=None)
            resid = window - F_aug @ coef

        if not np.isfinite(coef).all() or not np.isfinite(resid).all():
            continue

        resid_rows[i] = resid[-1]
        evr_rows[i] = pca.explained_variance_ratio_
        factor_rows[i] = F[-1]

        signal_date = returns.index[t]
        rolling_loadings[signal_date] = pd.DataFrame(
            pca.components_.T,
            index=tickers,
            columns=pc_cols,
        )

        if i == T - L - 1:
            loadings_last = pca.components_

    residuals = pd.DataFrame(resid_rows, index=dates, columns=tickers)
    evr = pd.DataFrame(evr_rows, index=dates, columns=pc_cols)
    factor_returns = pd.DataFrame(factor_rows, index=dates, columns=pc_cols)
    return residuals, evr, factor_returns, loadings_last, rolling_loadings


def compute_zscores(residuals: pd.DataFrame, M: int) -> pd.DataFrame:
    mu_e = residuals.rolling(M).mean().shift(1)
    sd_e = residuals.rolling(M).std(ddof=1).shift(1).replace(0.0, np.nan)
    return residuals.sub(mu_e).div(sd_e)


def load_clean_log_returns(
    processed_dir: Path = PROCESSED_DATA_DIR,
    min_col_obs_fraction: float = 0.95,
    max_abs_return: float = 0.5,
) -> pd.DataFrame:
    log_returns = pd.read_parquet(processed_dir / "log_returns.parquet")
    log_returns = log_returns.replace([np.inf, -np.inf], np.nan)
    log_returns = log_returns.mask(log_returns.abs() > max_abs_return)

    min_obs = int(min_col_obs_fraction * len(log_returns))
    good_cols = log_returns.columns[log_returns.notna().sum() >= min_obs]
    log_returns = log_returns[good_cols].dropna()
    return log_returns


def save_spec_artifacts(
    residuals: pd.DataFrame,
    zscores: pd.DataFrame,
    evr: pd.DataFrame,
    factor_returns: pd.DataFrame,
    loadings_last: np.ndarray | None,
    rolling_loadings: dict,
    L: int,
    K: int,
    M: int,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> dict[str, Path]:
    suffix = output_suffix(L, K, M)
    outputs = {
        "residuals": processed_dir / f"residual_returns{suffix}.parquet",
        "zscores": processed_dir / f"residual_zscores{suffix}.parquet",
        "evr": processed_dir / f"pca_explained_variance{suffix}.parquet",
        "factor_returns": processed_dir / f"pca_factor_returns{suffix}.parquet",
        "final_loadings": processed_dir / f"pca_loadings_final{suffix}.pkl",
        "rolling_loadings": processed_dir / f"pca_rolling_loadings{suffix}.pkl",
    }

    residuals.to_parquet(outputs["residuals"])
    zscores.to_parquet(outputs["zscores"])
    evr.to_parquet(outputs["evr"])
    factor_returns.to_parquet(outputs["factor_returns"])

    with open(outputs["final_loadings"], "wb") as fh:
        pickle.dump(
            {
                "L": L,
                "k": K,
                "M": M,
                "loadings": loadings_last,
                "components": [f"PC{j + 1}" for j in range(K)],
                "tickers": list(residuals.columns),
                "date": str(residuals.index[-1].date()),
                "timing_convention": "indexed_by_first_tradable_date",
            },
            fh,
        )

    with open(outputs["rolling_loadings"], "wb") as fh:
        pickle.dump(
            {
                "L": L,
                "k": K,
                "M": M,
                "loadings_by_date": rolling_loadings,
                "timing_convention": "indexed_by_first_tradable_date",
            },
            fh,
        )

    return outputs


def save_spec_outputs(
    log_returns: pd.DataFrame,
    L: int,
    K: int,
    M: int,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> dict[str, Path]:
    residuals, evr, factor_returns, loadings_last, rolling_loadings = rolling_pca_residuals(
        log_returns,
        L=L,
        k=K,
    )
    zscores = compute_zscores(residuals, M=M)
    return save_spec_artifacts(
        residuals=residuals,
        zscores=zscores,
        evr=evr,
        factor_returns=factor_returns,
        loadings_last=loadings_last,
        rolling_loadings=rolling_loadings,
        L=L,
        K=K,
        M=M,
        processed_dir=processed_dir,
    )


@app.command()
def generate_spec(
    L: int = typer.Option(..., help="PCA estimation window"),
    K: int = typer.Option(..., help="Number of PCA factors"),
    M: int = typer.Option(..., help="Residual z-score lookback"),
) -> None:
    log_returns = load_clean_log_returns()
    outputs = save_spec_outputs(log_returns, L=L, K=K, M=M)
    logger.success(f"Saved spec {output_suffix(L, K, M)}")
    for path in outputs.values():
        logger.info(path)


@app.command()
def generate_reduced_grid() -> None:
    log_returns = load_clean_log_returns()
    L_grid = [63, 126, 252]
    K_grid = [5, 10, 15]
    M_grid = [20]

    for L in L_grid:
        for K in K_grid:
            for M in M_grid:
                logger.info(f"Generating {output_suffix(L, K, M)}")
                save_spec_outputs(log_returns, L=L, K=K, M=M)

    logger.success("Reduced robustness grid generated.")


if __name__ == "__main__":
    app()
