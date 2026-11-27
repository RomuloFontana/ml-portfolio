# Bike Sharing Demand Prediction

## Objective
Predict hourly bicycle rental demand using weather and calendar features to support demand planning.

## Dataset
[UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset): 17,379 hourly observations from 2011–2012. The target is `cnt`. The variables `casual` and `registered` were excluded because their sum equals the target.

## Methodology
- Exploratory data analysis, feature selection and leakage prevention.
- Chronological 80/20 train/test split without shuffling.
- Five-fold time-series cross-validation using MAE.
- Comparison of DummyRegressor, linear regression, SGD and Random Forest.
- Evaluation of Random Forest on the held-out chronological test set.

## Results
| Model | Cross-validation MAE (mean ± standard deviation) |
| --- | ---: |
| DummyRegressor | 138.08 ± 30.75 |
| LinearRegression | 109.15 ± 23.26 |
| SGDRegressor (scaled) | 108.25 ± 23.12 |
| SGDRegressor (unscaled, no intercept) | 112.51 ± 21.33 |
| RandomForestRegressor | **52.29 ± 13.76** |

Random Forest reduced validation MAE by approximately 62.1% relative to the baseline. Test MAE: **44.99**; test RMSE: **69.56** rentals per hour.

## Reproduce
Open [the notebook](bike_sharing.ipynb) in Jupyter or Google Colab. Dependencies: Python, pandas, matplotlib, scikit-learn and ucimlrepo. The notebook retrieves the dataset from UCI.

## Limitations
Educational prototype; not deployed to production. An exploratory MLP experiment did not converge within 500 iterations. Future work includes cyclical time features, hyperparameter tuning and evaluation on fresh chronological data.
