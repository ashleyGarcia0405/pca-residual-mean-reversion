# pca-residual-mean-reversion

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

PCA residual mean reversion strategy implementation.

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