# Healthcare Patient Intelligence System

## Project Overview
End-to-end machine learning project on a healthcare dataset 
of 50,000 patients covering five prediction problems across 
classification and regression.

## Dataset
- 50,000 rows, 32 columns
- Synthetic dataset simulating real hospital EHR data
- Challenges handled: missing values, class imbalance, skewed targets, 
  outliers, multicollinearity

## Notebooks
- 01_eda.ipynb — Exploratory data analysis, null handling, outlier detection
- 02_readmission_prediction.ipynb — Binary classification, 15% imbalance, SMOTE vs class_weight
- 03_mortality_prediction.ipynb — Extreme imbalance (5.7%), SMOTE, threshold tuning
- 04_diagnosis_classification.ipynb — 5-class multiclass classification
- 05_length_of_stay.ipynb — Regression, log transform evaluation
- 06_treatment_cost.ipynb — Regression, log transform applied, skew=1.16

## Results
| Target | Best Model | Score |
|--------|-----------|-------|
| Readmission (30-day) | RF + class_weight | F1=0.27, Recall=0.94 (threshold=0.4) |
| Mortality Risk | XGBoost + scale_pos_weight | F1=0.15, Recall=0.52 |
| Diagnosis Category | Random Forest | Macro F1=0.33 |
| Length of Stay | XGBoost | R²=0.457, RMSE=1.47 days |
| Treatment Cost | XGBoost | R²=0.358, RMSE=$8,033 |

## Key Findings
- Weak feature signal limits all models — no single feature strongly 
  predicts any target
- Admission type (Emergency) is the strongest signal across multiple targets
- Real improvement requires richer clinical data: vital signs trends, 
  medication history, clinical notes

## Tech Stack
Python, scikit-learn, XGBoost, imbalanced-learn, pandas, numpy, 
matplotlib, seaborn

## How to Run
pip install -r requirements.txt ->in bash/terminal

Run notebooks in order: 01 → 06
Models will be saved to /models directory (not included in repo due to file size)