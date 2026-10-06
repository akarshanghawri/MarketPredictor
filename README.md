# Market Direction Prediction System

A time-series ML pipeline that predicts next-day price direction for RELIANCE (NSE) using engineered technical indicators and an XGBoost classifier, validated with walk-forward testing to avoid overfitting to a single lucky split.

## Overview

This project builds a full data-to-model pipeline for a stock direction prediction task:

- **Data**: ~5 years of daily OHLCV for RELIANCE (Sep 2021 – Aug 2026), sourced via NSE's historical-data portal, plus Nifty50 index data for relative-strength features.
- **Features**: ~30 engineered indicators — moving averages (SMA/EMA), MACD, RSI, Bollinger Bands, volatility, volume, and Nifty-relative features (`RELATIVE_STRENGTH`, `NIFTY_RSI_14`, etc.).
- **Model**: XGBoost classifier predicting binary next-day direction (`LABEL`) and 5-day direction (`LABEL_5D`).
- **Validation**: Walk-forward (expanding-window) validation across 5 out-of-sample folds, to check whether an apparent edge survives testing on multiple time periods rather than one split.

## Why walk-forward validation

An initial single chronological train/test split showed a modest edge on 1-day direction (53.59% vs. 49.37% naive baseline). Because that number came out of roughly ten experiments run in one session, it wasn't trustworthy on its own — with enough tries, some will look good by chance. Walk-forward validation re-tests the *same frozen model config* across 5 separate time windows to check whether the edge holds up out-of-sample, rather than reflecting one favorable split.

## Results

| Target | Pooled accuracy | Pooled naive baseline | z-score | Verdict |
|---|---|---|---|---|
| 1-day (`LABEL`) | 54.33% | 50.00% | ≈2.12 | Edge holds across all 5 folds — small but real |
| 5-day (`LABEL_5D`) | 51.05% | 50.21% | 0.26 | No distinguishable signal — noise |

The 1-day model beat the naive baseline in every one of the 5 folds (spanning March 2024 – August 2026), with the edge ranging from roughly 1.7pp to 10pp per fold. This consistency across genuinely different market periods is stronger evidence than the pooled z-score alone.

**Honest caveats:**
- A ~4.3 percentage point edge is real but modest — not close to a tradeable signal once transaction costs are considered.
- z≈2.12 clears the significance bar but isn't deep past it; a different window size or fold count could shift the result.
- Folds share overlapping training data by construction (expanding window), so they aren't fully independent experiments.
- **Feature importance is diffuse and unstable** — across folds, importance scores ranged narrowly (0.03–0.06 across 21 features, no dominant feature), and the top-ranked feature changed in every single fold (`CLOSE_TO_SMA20`, `RELATIVE_STRENGTH`, `HIGH_LOW_RANGE`, `RSI_14`, `VOLUME_CHANGE` — one per fold). This suggests the edge comes from many weakly correlated, partially redundant indicators rather than one robust, interpretable driver.

### Model selection

The XGBoost model above (54.33% pooled accuracy, z≈2.12) is the model used downstream (API, deployment). An LSTM was also tried as a comparison — see below — but did not outperform it, so it is not used in production.

## LSTM experiment (comparison, not used in production)

To test whether sequence modeling could find structure beyond the hand-crafted technical features, an LSTM was trained on 10-day sequences of the top 15 features (by importance averaged across the walk-forward folds above).

**First attempt overfit badly**: a 100-unit LSTM trained for 50 fixed epochs with no validation monitoring reached ~80% training accuracy but only ~49% test accuracy — essentially memorization with no generalization.

**Fixed by**: shrinking the model (32 units) and adding early stopping on validation loss, so training halts once the model stops generalizing rather than running for a fixed number of epochs.

**Result after the fix**: train and validation accuracy converged closely (~50–56%), and test accuracy landed at ~49–54% across two train/test ratios — statistically indistinguishable from chance, and no better than the naive baseline.

**Conclusion**: at this dataset size (~1,200 daily rows), the LSTM did not find sequential structure beyond what the engineered technical indicators already capture. This is a legitimate negative result rather than a failed experiment — it suggests more data would likely be needed for sequence modeling to add value here, and the XGBoost model remains the stronger validated option.

## Project structure

```
market-direction-prediction/
├── data/
│   ├── raw/              # Raw RELIANCE OHLCV CSVs from NSE
│   ├── raw_nifty/        # Raw Nifty50 index CSVs
│   └── processed/        # Merged, cleaned, feature-engineered data
├── src/
│   ├── merge_data.py         # Merges/cleans RELIANCE OHLCV CSVs, adjusts for bonus issue
│   ├── merge_nifty_data.py   # Merges Nifty50 index data into main dataset
│   ├── features.py           # Feature engineering (~30 columns)
│   ├── train_baseline.py     # Baseline XGBoost model (single chronological split)
│   ├── walk_forward.py       # Walk-forward validation (5-fold expanding window)
│   └── train_lstm.py         # LSTM comparison experiment (chance-level, not used in production)
└── notebooks/
    └── eda.ipynb          # Exploratory data analysis
```

## Methodology notes

- **Chronological splits only** — never random shuffle, to avoid look-ahead bias.
- **Split-adjustment** — RELIANCE's Oct 2024 1:1 bonus issue is auto-detected (>30% single-day price drop) and pre-event prices are scaled accordingly.
- **Raw price levels excluded from features** — only normalized/derived indicators (e.g. `CLOSE_TO_SMA_20` rather than raw `SMA_20`) are used, since raw price levels can create an illusory apparent edge unrelated to genuine predictive signal.
- **Frozen configuration during validation** — the walk-forward stage uses the exact same features and hyperparameters (XGBoost defaults) as the original result, with no tuning between folds, to keep the test honest.

## Roadmap

- [x] Data pipeline (RELIANCE + Nifty50 merge, cleaning, bonus adjustment)
- [x] Feature engineering (~30 indicators)
- [x] Baseline XGBoost model + walk-forward validation
- [x] LSTM sequence model — tried as a comparison; converged to chance-level accuracy after fixing overfitting (see LSTM experiment section); XGBoost remains the production model
- [ ] Flask API for serving predictions
- [ ] Celery + Redis for scheduled retraining
- [ ] Docker containerization
- [ ] Cloud deployment

## Disclaimer

This is a portfolio/educational project. Nothing here constitutes financial advice, and the model's accuracy is not sufficient for live trading without further work (cost modeling, risk management, and a stronger edge).