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

Download prices from Yahoo Finance, then preprocess them:

```bash
python data/download.py
python data/prepare.py
```

`data/universe.json` fixes the paper's assets and dates. Prices go to `data/raw/`
and model inputs to `data/processed/`; neither is included in Git.
New Yahoo downloads may differ from the original historical snapshot.

## Run the main experiments

```bash
bash reproduce.sh
```

The script runs the main experiments except Historic and PFL SD-relaxation on
S&P 500, which are skipped because of computational cost.
Results are saved in `outputs/paper/RESULTS.md` and `outputs/paper/tables/`.

The skipped cases and additional experiments remain available through `run.py`.
Use `python run.py --help` for commands and `python run.py evaluate --help` for evaluation options.
