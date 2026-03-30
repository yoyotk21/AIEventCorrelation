# AI Event Correlation

Detects correlations between Polymarket prediction markets and identifies mispriced parlays.

## What It Does

Builds an N×N **predictivity matrix** across resolved Polymarket contracts. Each cell `M[i][j]` scores how strongly market `i`'s trading behavior predicts market `j`'s outcome. High-scoring pairs are likely mispriced as independent when sold as parlays.

Weights for the 7 features are optimized via **hill climbing with simulated annealing**, validated on a held-out test set.

## Project Structure

```
data_collection.py      — Fetches data from Polymarket APIs, saves 3 CSVs
models.py               — Pydantic models: Market, PipelineConfig, result types
data_loader.py          — Loads CSVs into a DataBundle (markets + price/trade DataFrames)

features/
  base.py               — BaseFeature abstract class + normalize()
  same_category.py      — F4: binary same-category flag
  tag_jaccard.py        — F5: Jaccard similarity of tag sets
  resolution_proximity.py — F6: exp(-|days between resolution| / 30)
  temporal_overlap.py   — F7: fraction of shorter market's life that overlapped
  price_correlation.py  — F1: absolute Pearson r on early 40% of price overlap
  volume_spike.py       — F3: spike-day co-occurrence ratio (asymmetric)
  granger_causality.py  — F2: Granger causality F-statistic (asymmetric)

evaluation.py           — Train/test split, ground truth matrix, baselines
optimizer.py            — Hill climbing + simulated annealing weight search
analysis.py             — Mispricing table + stratified quantile analysis
pipeline.py             — Orchestrator: runs the full pipeline end to end
```

## How to Run

**1. Collect data** (requires internet, takes a few minutes):
```bash
python3 data_collection.py
```
Produces `markets.csv`, `prices_daily.csv`, `trades_daily.csv`.

**2. Run the pipeline:**
```bash
python3 pipeline.py
```
Outputs optimized weights, test-set MSE, Pearson r vs baselines, and a mispricing table.

## The 7 Features

| # | Name | Symmetric | Data Source |
|---|------|-----------|-------------|
| F1 | Price correlation (early window) | Yes | prices_daily.csv |
| F2 | Granger causality | **No** | prices_daily.csv |
| F3 | Volume spike co-occurrence | **No** | trades_daily.csv |
| F4 | Same category | Yes | markets.csv |
| F5 | Tag Jaccard similarity | Yes | markets.csv |
| F6 | Resolution date proximity | Yes | markets.csv |
| F7 | Temporal overlap ratio | Yes | markets.csv |

Each feature produces a raw N×N matrix. All 7 are min-max normalized to `[0, 1]`, then combined: `M = w1*F1 + ... + w7*F7`.

## Configuration

All parameters live in `PipelineConfig` in `models.py`:

```python
PipelineConfig(
    train_cutoff="2024-10-01",  # time-based train/test split
    min_overlap_days=7,         # minimum shared trading days for a valid pair
    num_restarts=50,            # hill climbing random restarts
    max_iterations=5000,        # iterations per restart
)
```

## Dependencies

```
pip install pydantic statsmodels scipy pandas numpy requests
```
