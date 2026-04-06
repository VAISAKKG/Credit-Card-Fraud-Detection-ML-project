# 📄 Project Report — Credit Card Fraud Detection

**Author:** Vaisak  
**Domain:** Machine Learning | Financial Analytics  
**Tools:** Python, Scikit-learn, Streamlit  
**Dataset:** Kaggle — Credit Card Fraud Detection (ULB)

---

## 1. Introduction

### 1.1 Background

Credit card fraud causes billions of dollars in losses globally each year. Financial institutions need automated, reliable systems that can flag suspicious transactions in real time without disrupting legitimate customer activity. Machine learning offers a scalable solution — models trained on historical transaction patterns can generalise to detect novel fraud attempts.

### 1.2 Problem Statement

Build a binary classification system that accurately identifies fraudulent credit card transactions from a highly imbalanced dataset, and deploy the model as a usable web application for batch prediction.

### 1.3 Objectives

- Explore and understand the distribution of fraud vs. legitimate transactions
- Train and compare classification models suitable for imbalanced data
- Select the best model using an appropriate evaluation metric (ROC-AUC)
- Save and deploy the model via a Streamlit web interface

---

## 2. Dataset Description

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
**File:** `creditcard.csv`

| Property | Value |
|----------|-------|
| Total Records | 284,807 |
| Features | 31 |
| Missing Values | None |
| Target Column | `Class` (0 = Legitimate, 1 = Fraud) |

### 2.1 Features

| Feature | Description |
|---------|-------------|
| `Time` | Seconds elapsed since the first transaction in the dataset |
| `V1–V28` | Anonymised features derived via PCA (original features confidential) |
| `Amount` | Transaction value in currency units |
| `Class` | Binary target label |

### 2.2 Class Imbalance

| Class | Count | Share |
|-------|-------|-------|
| Legitimate (0) | 284,315 | 99.83% |
| Fraud (1) | 492 | 0.17% |

The dataset is severely imbalanced with a fraud rate of just **0.17%**. This rules out accuracy as a reliable metric and requires careful model selection and evaluation strategy.

---

## 3. Exploratory Data Analysis

### 3.1 Data Quality
- Zero missing values confirmed across all 31 columns
- No duplicate handling required at this stage
- All features are numeric (float64)

### 3.2 Class Distribution
A countplot confirmed the extreme imbalance — legitimate transactions vastly outnumber fraudulent ones. This was a key driver of metric selection (see Section 5).

### 3.3 Transaction Amount Distribution
A histogram with KDE overlay showed that most transactions cluster at lower amounts. High-value transactions are rare across both classes.

---

## 4. Methodology

### 4.1 Feature and Target Separation

```python
X = credit_card_data.drop("Class", axis=1)
y = credit_card_data["Class"]
```

All 30 features (Time, V1–V28, Amount) were used. No manual feature selection was applied given the PCA-transformed nature of V1–V28.

### 4.2 Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

- **80% training / 20% testing**
- `stratify=y` ensures the fraud rate is preserved in both splits
- `random_state=42` for full reproducibility

| Split | Records |
|-------|---------|
| Training set | 227,845 |
| Test set | 56,962 |

### 4.3 Preprocessing

`StandardScaler` was applied inside a `sklearn Pipeline` for each model. This ensures the scaler is fitted only on training data — preventing data leakage into the test set.

```python
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", model)
])
```

---

## 5. Model Training & Evaluation

### 5.1 Models Compared

| Model | Notes |
|-------|-------|
| Logistic Regression | Linear baseline; `max_iter=1000` |
| Decision Tree | Non-linear; default hyperparameters |

### 5.2 Evaluation Metrics

Given the class imbalance, the following metrics were tracked:

| Metric | Reason |
|--------|--------|
| **ROC-AUC** | Primary metric — measures ranking ability regardless of threshold |
| Accuracy | Reported but not used for selection (misleading on imbalanced data) |
| Precision | Share of predicted fraud that is actually fraud |
| Recall | Share of actual fraud cases correctly caught |
| F1 Score | Harmonic mean of Precision and Recall |

> A model predicting "legitimate" for every transaction achieves 99.83% accuracy while catching zero fraud. ROC-AUC is immune to this failure mode.

### 5.3 Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 99.91% | 82.67% | 63.27% | 71.68% | **0.9605** |
| Decision Tree | 99.91% | 73.47% | 73.47% | 73.47% | 0.8671 |

### 5.4 Confusion Matrix (Best Model — Logistic Regression)

|  | Predicted: Legit | Predicted: Fraud |
|--|-----------------|-----------------|
| **Actual: Legit** | True Negatives | False Positives |
| **Actual: Fraud** | False Negatives | True Positives |

The confusion matrix heatmap confirmed the model reliably identifies the majority of fraud cases while maintaining a low false positive rate.

---

## 6. Best Model Selection

**Winner: Logistic Regression**

Selected on the basis of the highest ROC-AUC score of **0.9605**, meaning the model correctly ranks a fraudulent transaction above a legitimate one 96% of the time.

The model pipeline (scaler + classifier) was serialised:

```python
joblib.dump(best_model, "fraud_model.pkl")
```

---

## 7. Streamlit Web Application

### 7.1 Overview

`app.py` wraps the saved model in a user-friendly web interface for batch inference.

### 7.2 Application Flow

```
User uploads CSV
       ↓
App previews data
       ↓
User clicks "Run Fraud Detection"
       ↓
Model predicts each transaction
       ↓
Results displayed with Fraud Probability column
       ↓
Summary: Fraud count vs. Legitimate count
```

### 7.3 Key Features

- Accepts CSV files in the same column format as the training dataset
- Automatically drops `Class` column if present in the upload
- Returns both a binary prediction (`0`/`1`) and a continuous fraud probability score per transaction
- Displays a summary panel showing total fraud vs. legitimate transactions detected
- Error handling for malformed or incompatible uploads

### 7.4 Running the App

```bash
streamlit run app.py
```

Access at: `http://localhost:8501`

---

## 8. Key Findings

- The dataset's extreme class imbalance (0.17% fraud) makes this a challenging but realistic detection problem
- Logistic Regression outperformed Decision Tree on ROC-AUC despite similar accuracy scores — demonstrating why metric selection matters for imbalanced problems
- The Pipeline architecture ensured no preprocessing leakage between train and test sets
- The saved model generalises well, correctly identifying fraud with 96% ranking confidence on held-out data

---

## 9. Limitations & Future Work

| Limitation | Suggested Improvement |
|------------|-----------------------|
| Only 2 models compared | Extend to Random Forest, XGBoost, LightGBM |
| No resampling used | Apply SMOTE or class weighting to improve Recall |
| Default hyperparameters | Add GridSearchCV or RandomizedSearchCV tuning |
| No threshold optimisation | Tune decision threshold to balance Precision vs. Recall |
| Static model file | Add model retraining capability in the app |

---

## 10. Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.x | Core language |
| Pandas | Latest | Data loading and manipulation |
| NumPy | Latest | Numerical operations |
| Scikit-learn | Latest | ML models, pipelines, metrics |
| Matplotlib | Latest | Visualisation |
| Seaborn | Latest | Statistical plots |
| Joblib | Latest | Model serialisation |
| Streamlit | Latest | Web application deployment |

---

## 11. Project Files

| File | Description |
|------|-------------|
| `Credit_Card_Fraud_Detection.ipynb` | End-to-end ML pipeline — EDA, training, evaluation, model export |
| `app.py` | Streamlit web application for batch fraud prediction |
| `fraud_model.pkl` | Serialised Logistic Regression pipeline (scaler + model) |
| `creditcard.csv` | Source dataset — 284,807 transactions, 31 features |
| `README.md` | Quick-start guide and project summary |
| `PROJECT_REPORT.md` | This document — detailed methodology and findings |

---

*Report prepared for GitHub portfolio submission.*
