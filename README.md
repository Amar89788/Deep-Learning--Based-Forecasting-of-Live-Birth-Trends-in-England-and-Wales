# Deep Learning-Based Forecasting of Live Birth Trends in England and Wales

This project forecasts the UK live birth rate using deep learning. It builds and compares
three neural network models - GRU, LSTM and a 1D CNN - on 64 years of annual data, to
answer one question: how effectively can deep learning forecast long-term birth trends
from demographic time-series data?

## Research question

How effectively can deep learning models forecast live birth counts from long-term
demographic time-series data?

## Dataset

- **Source:** https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages
- **Scope:** United Kingdom, 1960–2023 (64 annual observations)
- **Quality:** no missing values, no duplicate years
- **Licence:** Creative Commons Attribution 4.0 (CC BY 4.0)

The raw CSV covers all countries; the code filters it to the United Kingdom. If
`birthrates.csv` is not present, the notebook downloads the same indicator directly from
the World Bank API as a fallback.

## Method

1. **Load and clean** - filter the CSV to the UK, keep Year and Birth Rate, sort chronologically and set Year as the index.
2. **EDA** - trend line, distribution (histogram + KDE), box plot, lag-1 plot and autocorrelation (ACF) plot.
3. **Split** - chronological, no shuffling: 1960–2010 for training (51 points), 2011–2023 for testing (13 points).
4. **Differencing** - the models forecast the year-on-year *change*, not the level. This is the key step: 2023 is the lowest value in the whole series, and a scaled network cannot predict below its training range, so a level-based model bends the forecast the wrong way. Predictions are reconstructed as the previous actual level plus the predicted change.
5. **Scaling** - a MinMaxScaler is fitted on the training-period changes only (no leakage), then applied to the full series.
6. **Windowing** - 5-year sliding windows (the previous 5 changes predict the next).
7. **Modelling** - GRU, LSTM and 1D CNN, each trained as a 5-seed ensemble (seeds 42–46) with predictions averaged.
8. **Evaluation** - MAE, RMSE, R² and MAPE on the 2011–2023 test set.
9. **Forecast** - a recursive multi-step forecast to 2030.

## Models

All models use a 5-year input window and are trained with the Adam optimiser (learning
rate 0.001), MSE loss, batch size 4, up to 500 epochs, and early stopping (patience 60,
restore best weights). Each is a 5-seed ensemble.

- **GRU** - one GRU layer (64 units, tanh) → Dropout(0.1) → Dense(1)
- **LSTM** - one LSTM layer (64 units, tanh) → Dropout(0.1) → Dense(1)
- **1D CNN** - Conv1D (64 filters, kernel size 2, ReLU) → Flatten → Dense(1)

## Results (test set 2011–2023)

| Model         | MAE    | RMSE   | R²     | MAPE   |
|---------------|--------|--------|--------|--------|
| **CNN** (best)| 0.1957 | 0.2776 | 0.9176 | 1.73%  |
| LSTM          | 0.2380 | 0.3147 | 0.8942 | 2.08%  |
| GRU           | 0.2414 | 0.3176 | 0.8922 | 2.12%  |

The 1D CNN performed best on every metric. All three models reached R² close to 0.9 on a
strictly out-of-sample 13-year test, with average errors around 0.2 births per 1,000. The
2024–2030 forecast projects the birth rate rising gently from about 10.0 to 10.7 births
per 1,000.

## Reproducibility

Results are fully reproducible: all random seeds are locked (Python, NumPy and TensorFlow
via `tf.keras.utils.set_random_seed`) and TensorFlow op-determinism is enabled, with an
ensemble seeded 42–46. Running the notebook top to bottom reproduces the numbers above.
               
## Tools

Python · TensorFlow / Keras · scikit-learn · statsmodels · pandas · NumPy · Matplotlib · Seaborn

## Author

Amarnadh Kandimalla — Student ID: 23075810
MSc Data Science, University of Hertfordshire
