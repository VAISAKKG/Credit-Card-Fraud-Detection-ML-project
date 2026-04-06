# Credit-Card-Fraud-Detection-ML-project
Machine Learning | Financial Analytics |Python, Scikit-learn, Streamlit  

A machine learning project to detect fraudulent credit card transactions using classification algorithms, deployed as an interactive web application with Streamlit.

---

## 📌 Project Overview

Credit card fraud is a critical problem in financial systems. This project builds and compares classification models on a highly imbalanced real-world dataset to accurately flag fraudulent transactions — minimising both false negatives (missed fraud) and false positives (legitimate transactions blocked).

The best-performing model is saved and served through a Streamlit web app that accepts CSV uploads and returns fraud predictions with confidence scores.

---

## 🗂️ Repository Structure

```
credit-card-fraud-detection/
│
├── Credit_Card_Fraud_Detection.ipynb   # Full ML pipeline notebook
├── app.py                              # Streamlit web application
├── fraud_model.pkl                     # Saved best model (Logistic Regression)
├── creditcard.csv                      # Dataset (284,807 transactions)
└── README.md                           # Project documentation
```

---

## 📁 Dataset

**File:** `creditcard.csv`  
**Records:** 284,807 transactions  
**Features:** 31 columns

| Column | Description |
|--------|-------------|
| `Time` | Seconds elapsed between this and the first transaction |
| `V1–V28` | PCA-transformed anonymised features |
| `Amount` | Transaction amount |
| `Class` | Target — `0` = Legitimate, `1` = Fraudulent |

**Class Distribution:**

| Class | Count | Percentage |
|-------|-------|------------|
| Legitimate (0) | 284,315 | 99.83% |
| Fraud (1) | 492 | 0.17% |

> ⚠️ The dataset is highly imbalanced. ROC-AUC was used as the primary evaluation metric rather than accuracy.

---

## 🔬 ML Pipeline

### 1. Data Loading & Validation
- Loaded 284,807 transaction records
- Confirmed zero missing values across all 31 columns
- Visualised class distribution and transaction amount distribution

### 2. Feature Engineering
- Dropped the `Class` target column to form feature matrix `X`
- Applied `StandardScaler` inside a `Pipeline` to prevent data leakage

### 3. Train-Test Split
- 80/20 split with `stratify=y` to preserve class imbalance ratio
- `random_state=42` for reproducibility

### 4. Models Trained

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 99.91% | 82.67% | 63.27% | 71.68% | **0.9605** |
| Decision Tree | 99.91% | 73.47% | 73.47% | 73.47% | 0.8671 |

### 5. Best Model
**Logistic Regression** was selected based on highest ROC-AUC score (0.9605).

> ROC-AUC was chosen because it measures the model's ability to rank fraud above legitimate transactions — the most relevant metric for imbalanced classification problems.

### 6. Model Export
The best model pipeline (scaler + classifier) was serialised using `joblib` and saved as `fraud_model.pkl`.

---

## 🖥️ Streamlit App

The `app.py` file provides an interactive web interface for real-time batch predictions.

**Features:**
- Upload any CSV file in the same format as the training dataset
- Preview uploaded data before running predictions
- Run fraud detection with one click
- View results with per-transaction fraud probability scores
- Summary count of fraud vs. legitimate transactions detected

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn streamlit joblib matplotlib seaborn
```

### Run the Notebook

Open and run `Credit_Card_Fraud_Detection.ipynb` end-to-end. This will:
1. Train and evaluate both models
2. Select the best model
3. Save `fraud_model.pkl`

### Launch the Web App

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

### Make a Prediction

1. Prepare a CSV with the same columns as `creditcard.csv` (excluding `Class`)
2. Upload it in the app
3. Click **Run Fraud Detection**
4. Review predictions and fraud probability scores

---

## 📊 Why ROC-AUC Over Accuracy?

With only 0.17% fraud cases, a model predicting "legitimate" for every transaction achieves 99.83% accuracy — but catches zero fraud. ROC-AUC evaluates how well the model **ranks** fraudulent transactions above legitimate ones regardless of threshold, making it the correct metric for this problem.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Core language |
| Pandas / NumPy | Data manipulation |
| Scikit-learn | ML models, pipelines, metrics |
| Matplotlib / Seaborn | Visualisation |
| Joblib | Model serialisation |
| Streamlit | Web application |

---

## 👤 Author

**Vaisak**  
Data Analyst | MSc Data Science  
*Skills: Python, Machine Learning, Streamlit, SQL, Tableau, Power BI*

---

## 📝 License

This project is intended for portfolio and educational purposes.  
Dataset source: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
