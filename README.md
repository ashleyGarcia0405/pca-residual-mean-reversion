# pca-residual-mean-reversion

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

PCA residual mean reversion strategy implementation.

## Phase H — asymmetric composite strategy

After 87 experiments (see `experiments/reports/FINAL_REPORT.md`), the integrated strategy delivers:

| Metric | Value |
|---|---|
| Sharpe gross | **+1.16** |
| Sharpe net @ 1 bps | +1.11 |
| Sharpe net @ 3 bps | **+1.00** |
| Sharpe net @ 5 bps | **+0.89** |
| Annual return @ 5 bps | **+3.67%** |
| Max drawdown @ 5 bps | -7.3% |
| Daily turnover | 0.085 |

Both halves of 2012-2026 deliver Sharpe ≈ +0.90 at 5 bps (13 of 15 full years profitable).

Three structural changes drive the result:

1. **Timing-aligned PCA residuals**. The legacy `rolling_pca_residuals` forward-shifts each residual by one day. The new `align="actual"` mode labels each residual at the date of its underlying return, so daily-rebalanced backtests no longer skip the bounce day.
2. **Composite signal**. Moderate-$z$ residuals (|$z$| ≈ 1–2) revert; extreme-$z$ residuals (|$z$| > 2.5) continue. Combining both regimes via `build_weights_asymmetric_composite` captures more alpha than either alone.
3. **Asymmetric sparse sizing**. Concentrating $k_{rev,long}=6$ names on the long side while spreading $k_{rev,short}=14$ on the short side dramatically outperforms symmetric portfolios (gross +1.16 vs +0.13). The opposite asymmetry collapses to negative gross.

Reproduce via `notebooks/6.0-ag-asymmetric-composite-strategy.ipynb`. Generate the timing-aligned PCA artifacts with:

```bash
python -m pca_residual_mean_reversion.residuals generate-spec --L 126 --K 10 --M 20 --align actual
```

## Project Organization

```
├── LICENSE
├── Makefile           <- Convenience commands like `make data` or `make train`
├── README.md
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- Documentation
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention: number (for ordering),
│                         creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-ag-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration and package metadata
│
├── references         <- Data dictionaries, manuals, and other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures
│
├── requirements.txt   <- Requirements file for reproducing the analysis environment
│
├── setup.cfg          <- Configuration file for flake8
│
└── pca_residual_mean_reversion   <- Source code for use in this project.
    │
    ├── __init__.py
    ├── config.py               <- Store useful variables and configuration
    ├── dataset.py              <- Scripts to download or generate data
    ├── features.py             <- Code to create features for modeling
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py
    │   └── train.py
    └── plots.py                <- Code to create visualizations
```

--------