# Ml-Assignment
# Polynomial Regression 


## Overview
This project implements polynomial regression to predict target values for two nonlinear regression datasets using Python and Scikit-learn.

## Models Used
- **Variant 1:** Degree-5 polynomial regression using Lasso-based feature selection and OLS refitting.
- **Variant 2:** Ridge-regularized polynomial regression combining degrees 10, 11, and 12.

## Results

| Metric | Variant 1 | Variant 2 |
|---|---|---|
| 5-Fold CV R² | 0.970343 | 0.993862 |
| 5-Fold CV MSE | 0.296301 | 0.248718 |

## Requirements
Python 3.x, NumPy, SciPy, Scikit-learn, Matplotlib and Joblib.

## How to Run
Install libs:

```bash
pip install numpy scipy scikit-learn matplotlib joblib
```

Place the training and test CSV files along with `sample_submission.csv` in the project folder.

Run:

```bash
python trainingcode.py --data-dir .
```

## Outputs
- `IMT2024085-pred_var1.csv`
- `IMT2024085-pred_var2.csv`

Each output contains 1,000 predictions in a single column named `y`.

