
# Concrete Compressive Strength Prediction

My first end-to-end machine learning project using Python and Scikit-learn.

## Project Overview

This project predicts the compressive strength of concrete (MPa) based on its material composition and curing age.

The dataset comes from the UCI Machine Learning Repository and contains 1,030 samples with 8 original input features.

## Methods

- Linear Regression (baseline model)
- Random Forest Regression
- Hyperparameter tuning
- 5-Fold Cross Validation
- Feature Engineering
- Error Analysis and Feature Importance

## Results

| Model | 5-Fold CV MAE |
|---|---|
| Linear Regression (8 features) | 8.422 MPa |
| Random Forest (8 features) | 3.777 MPa |
| Random Forest (9 features) | 3.647 MPa |

The original 8-feature Random Forest achieved a test MAE of 3.835 MPa.

A new feature, Water-to-Binder Ratio, was introduced to improve cross-validation performance.

## Project Files

- `main.py`: Model training and evaluation
- `predict.py`: Load the saved model and make predictions
- `RandomForestModel.joblib`: Trained Random Forest model
- `requirements.txt`: Python dependencies

## How to Run

Create and activate a Python virtual environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the prediction script:

```bash
python predict.py
```

## Dataset

UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength

## Disclaimer

This project is for educational purposes only. Predictions should not be used for engineering safety decisions.
