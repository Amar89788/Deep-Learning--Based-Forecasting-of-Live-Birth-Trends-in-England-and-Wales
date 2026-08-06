# Deep Learning-Based Forecasting of Live Birth Trends in England and Wales

This project forecasts the UK live birth rate using deep learning. It compares three
neural network models (GRU, LSTM and a 1D CNN) on 64 years of annual data and asks a
simple question: can deep learning actually forecast long-term birth trends better than
a naive "next year = this year" baseline?

Short answer: yes — but only when the data is prepared properly and every model is
benchmarked honestly.

## Research question

How effectively can deep learning models forecast live birth trends from long-term
demographic time-series data?

## Dataset

- **Source:** World Bank, World Development Indicators — *Birth rate, crude (per 1,000 people)*, indicator `SP.DYN.CBRT.IN`
- **Link:** https://data.worldbank.org/indicator/SP.DYN.CBRT.IN
- **Scope:** United Kingdom, 1960–2023 (64 annual observations)
- **Quality:** no missing values, no duplicate years
- **Licence:** Creative Commons Attribution 4.0 (CC BY 4.0)

The raw CSV covers all countries; the code filters it to the United Kingdom.

## What the code does

1. Loads the CSV and filters to the UK (Year + Birth Rate).
2. Runs EDA — trend line, distribution, box plot, lag-1 and ACF plots.
3. Splits the data chronologically: 1960–2010 for training, 2011–2023 for testing (no shuffling).
4. Applies **first-order differencing** — the key step. 2023 is the lowest value in the whole
   series, and a scaled network can't predict below its training range, so the models forecast
   the year-on-year *change* instead of the level.
5. Scales the differenced data (scaler fitted on the training set only — no leakage).
6. Builds 5-year sliding windows.
7. Trains three models — GRU, LSTM, 1D CNN — each as a **5-seed ensemble** for stability.
8. Evaluates one-step-ahead against a naive persistence baseline using MAE, RMSE, R² and MAPE.
9. Produces a recursive forecast to 2030.

## Results (test set 2011–2023, one-step-ahead)

| Model               | MAE   | RMSE  | R²    | MAPE  |
|---------------------|-------|-------|-------|-------|
| **1D CNN** (best)   | 0.196 | 0.278 | 0.918 | 1.73% |
| LSTM                | 0.238 | 0.315 | 0.894 | 2.08% |
| GRU                 | 0.241 | 0.318 | 0.892 | 2.12% |
