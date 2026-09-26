# RA-STPO

Code for **Regularization-Aligned Learning for Sparse Portfolio Optimization**.

## Install

Run from this directory:

```bash
conda create -n rastpo python=3.11 -y
conda activate rastpo
python -m pip install -r requirements.txt
```

## Prepare the data

`data/raw/` contains the fixed Yahoo Finance adjusted-price snapshot used in the paper.
A separate [Yahoo-derived Kaggle archive](https://www.kaggle.com/datasets/benjaminpo/finance-dataset/versions/47)
was used to cross-check selected historical records.

Preprocess the bundled prices:

```bash
python data/prepare.py
```

Model inputs are written to `data/processed/`.

## Run the main experiments

```bash
bash reproduce.sh
```

The script runs the main experiments except Historic and PFL SD-relaxation on
S&P 500, which are skipped because of computational cost.
Results are saved in `outputs/paper/RESULTS.md` and `outputs/paper/tables/`.

The skipped cases and additional experiments remain available through `run.py`.
Use `python run.py --help` for commands and `python run.py evaluate --help` for evaluation options.
