# Classification Experiments: MNIST and Titanic

Educational exercises inspired by Chapter 3 of Aurélien Géron's *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd edition).

## MNIST — Handwritten Digit Recognition
- Loaded MNIST from OpenML (60,000 training images and 10,000 test images).
- Configured a K-Nearest Neighbors classifier and implemented one-pixel image shifts in four directions using SciPy for data augmentation.
- The training and scoring cells are commented out in the saved notebook. No final accuracy or measured augmentation improvement is reported.

## Titanic — Passenger Survival Prediction
- Explored 891 labeled passenger records, including missing values and correlations.
- Engineered features from age, family size, passenger titles and cabin deck letters.
- Built numerical and categorical preprocessing pipelines with imputation, scaling and encoding using ColumnTransformer.
- Compared Random Forest, SVM, KNN and XGBoost with ten-fold cross-validation. Tuned KNN and XGBoost using GridSearchCV.

## Recorded Titanic Results
| Model | Mean CV accuracy |
| --- | ---: |
| XGBoost | **83.39%** |
| SVM | 82.49% |
| Random Forest | 81.26% |
| KNN | 80.92% |

These are exploratory cross-validation results, not independent test-set scores.

## Reproduce
Open [the notebook](classification_experiments.ipynb) in Jupyter or Google Colab. Use Python 3.12+ with numpy, pandas, scipy, matplotlib, scikit-learn and xgboost. MNIST is fetched from OpenML; Titanic is downloaded from the book's dataset URL.

The archive extraction cell uses Python's `filter='data'` for safer extraction. Original experimental outputs are retained.

## Limitations
Titanic preprocessing was fitted before cross-validation, potentially introducing data leakage. Hyperparameters were selected and assessed on the same training data. Future work: fit learned preprocessing within each CV fold, use nested cross-validation or an independent holdout, and complete MNIST training and evaluation. No production deployment was performed.
