# UPI Fraud Detection Using Machine Learning

A complete fraud detection system that uses **Logistic Regression**, **Random Forest**, and **XGBoost** models to identify fraudulent UPI transactions, served through a premium web dashboard.

## 🏗️ Project Structure

```
mini project/
├── data/
│   └── upi_transactions.csv          # Synthetic dataset (100K rows)
├── models/
│   ├── best_model.pkl                 # Best performing model
│   ├── scaler.pkl                     # Feature scaler
│   ├── label_encoders.pkl             # Categorical encoders
│   └── metrics.json                   # Evaluation metrics
├── static/
│   ├── css/style.css                  # Dashboard styles
│   ├── js/app.js                      # Dashboard logic
│   └── plots/                         # Generated charts
├── templates/
│   └── index.html                     # Dashboard HTML
├── generate_dataset.py                # Dataset generator
├── train_model.py                     # ML training pipeline
├── app.py                             # Flask web server
├── requirements.txt                   # Dependencies
└── README.md                          # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset
```bash
python generate_dataset.py
```

### 3. Train Models
```bash
python train_model.py
```

### 4. Launch Dashboard
```bash
python app.py
```
Then open **http://localhost:5000** in your browser.

## 📊 Dataset Features

| Feature | Description |
|---|---|
| `amount` | Transaction amount (₹) |
| `transaction_type` | P2P, P2M, Bill Payment, Recharge |
| `sender/receiver_bank` | Bank names |
| `sender_balance_before/after` | Balance changes |
| `device_type` | Android, iOS, Web |
| `sender/receiver_city` | City names |
| `upi_app` | GPay, PhonePe, Paytm, etc. |
| `is_fraud` | Target (0=Legit, 1=Fraud) |

## 🤖 ML Pipeline

1. **Feature Engineering** — Temporal features, balance ratios, log transforms
2. **SMOTE** — Handles class imbalance (2.5% fraud rate)
3. **3 Models** — Logistic Regression, Random Forest, XGBoost
4. **Evaluation** — Accuracy, Precision, Recall, F1, AUC-ROC, Confusion Matrices

## 🛡️ Fraud Patterns Detected

- Large transactions at unusual hours (midnight–4 AM)
- Rapid burst transactions (velocity fraud)
- Account balance draining attempts
- Unusually high amounts relative to history
- Cross-city anomalies

## 📝 Tech Stack

- **Python** — pandas, numpy, scikit-learn, XGBoost, imbalanced-learn
- **Visualization** — matplotlib, seaborn
- **Web** — Flask, HTML5, CSS3, JavaScript
