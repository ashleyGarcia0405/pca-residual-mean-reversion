"""Phase D: regenerate PCA artifacts with the residual labeled at its actual date (t-1)
instead of forward-shifted to t. Output suffix: _L{L}_K{K}_M{M}_t0
"""
import pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
L, K, M = 126, 10, 20
SUFFIX = f"_L{L}_K{K}_M{M}_t0"

log_returns = pd.read_parquet(PROCESSED / "log_returns.parquet")
log_returns = log_returns.replace([np.inf, -np.inf], np.nan)
log_returns = log_returns.mask(log_returns.abs() > 0.5)
min_obs = int(0.95 * len(log_returns))
good = log_returns.columns[log_returns.notna().sum() >= min_obs]
log_returns = log_returns[good].dropna()

T, N = log_returns.shape
tickers = log_returns.columns
# TIMING FIX: label residual at returns.index[t-1] instead of returns.index[t]
dates = log_returns.index[L - 1 : T - 1]
R = log_returns.values

resid_rows = np.full((T - L, N), np.nan)
factor_rows = np.full((T - L, K), np.nan)
evr_rows = np.full((T - L, K), np.nan)
loadings_last = None
rolling_loadings = {}
pc_cols = [f"PC{j+1}" for j in range(K)]

for i, t in enumerate(tqdm(range(L, T), desc=f"PCA t0 L={L} K={K}")):
    window = R[t - L : t].copy()
    mu = window.mean(axis=0)
    sig = window.std(axis=0, ddof=1)
    sig = np.where(sig == 0, 1.0, sig)
    R_std = (window - mu) / sig
    pca = PCA(n_components=K)
    F = pca.fit_transform(R_std)
    F_aug = np.column_stack([np.ones(L), F])
    coef, _, _, _ = np.linalg.lstsq(F_aug, window, rcond=None)
    resid = window - F_aug @ coef
    if not np.isfinite(resid).all():
        continue
    resid_rows[i] = resid[-1]
    evr_rows[i] = pca.explained_variance_ratio_
    factor_rows[i] = F[-1]
    # rolling_loadings still indexed by the trade date (t-1)
    rolling_loadings[log_returns.index[t - 1]] = pd.DataFrame(
        pca.components_.T, index=tickers, columns=pc_cols
    )
    if i == T - L - 1:
        loadings_last = pca.components_

residuals = pd.DataFrame(resid_rows, index=dates, columns=tickers)
evr = pd.DataFrame(evr_rows, index=dates, columns=pc_cols)
factor_returns = pd.DataFrame(factor_rows, index=dates, columns=pc_cols)

mu_e = residuals.rolling(M).mean().shift(1)
sd_e = residuals.rolling(M).std(ddof=1).shift(1).replace(0.0, np.nan)
zscores = residuals.sub(mu_e).div(sd_e)

residuals.to_parquet(PROCESSED / f"residual_returns{SUFFIX}.parquet")
zscores.to_parquet(PROCESSED / f"residual_zscores{SUFFIX}.parquet")
evr.to_parquet(PROCESSED / f"pca_explained_variance{SUFFIX}.parquet")
factor_returns.to_parquet(PROCESSED / f"pca_factor_returns{SUFFIX}.parquet")
with open(PROCESSED / f"pca_loadings_final{SUFFIX}.pkl", "wb") as fh:
    pickle.dump({"L": L, "k": K, "M": M, "loadings": loadings_last,
                 "components": pc_cols, "tickers": list(tickers),
                 "date": str(residuals.index[-1].date()),
                 "timing_convention": "indexed_by_actual_residual_date"}, fh)
with open(PROCESSED / f"pca_rolling_loadings{SUFFIX}.pkl", "wb") as fh:
    pickle.dump({"L": L, "k": K, "M": M, "loadings_by_date": rolling_loadings,
                 "timing_convention": "indexed_by_actual_residual_date"}, fh)
print(f"Wrote {SUFFIX} artifacts.")
