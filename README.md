# 💳 UPI Fraud Detection using Machine Learning

A full-stack web application that detects **fraudulent UPI transactions** using Machine Learning. The system analyzes transaction patterns — particularly **time-based anomalies** — by comparing a new transaction against a user's previous transaction history to flag suspicious activity in real time.

---

## 🎯 Features

- 🕵️ Detects suspicious transactions based on unusual timing patterns
- 📊 Compares incoming transactions against historical user transaction data
- 🤖 Machine Learning model to classify transactions as **Fraud** / **Genuine**
- 🌐 Full-stack web app with a clean, interactive UI
- 🗄️ MySQL database for storing and querying transaction records
- ⚡ Real-time fraud alerts based on model predictions

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python |
| Machine Learning | Python (Scikit-learn / Pandas / NumPy) |
| Database | MySQL |

---

## 📂 Project Structure

```
├── frontend/                # UI files
│   ├── index.html
│   ├── style.css
│   └── script.js
├── backend/                 # Python backend + ML logic
│   ├── app.py                # Main server/API file
│   ├── model/
│   │   ├── train_model.py    # Model training script
│   │   └── fraud_model.pkl   # Trained model file
│   ├── database/
│   │   └── db_connection.py  # MySQL connection logic
│   └── requirements.txt
├── dataset/                 # Sample/training transaction dataset
│   └── transactions.csv
├── screenshots/             # App screenshots/demo images
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- MySQL Server installed and running
- pip (Python package manager)
- A modern web browser

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```

2. Install Python dependencies
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. Set up the MySQL database
   ```sql
   CREATE DATABASE upi_fraud_detection;
   -- Import the schema/sample data if provided
   SOURCE database/schema.sql;
   ```

4. Update database credentials in `db_connection.py`
   ```python
   DB_HOST = "localhost"
   DB_USER = "your_username"
   DB_PASSWORD = "your_password"
   DB_NAME = "upi_fraud_detection"
   ```

5. Run the backend server
   ```bash
   python app.py
   ```

6. Open `frontend/index.html` in your browser (or the URL shown by the server)

---

## 🧠 How It Works

1. Every incoming UPI transaction (amount, time, sender/receiver, etc.) is logged.
2. The system retrieves the user's **previous transaction history** from MySQL.
3. Time-based features are engineered — e.g., time gap since last transaction, frequency of transactions in unusual hours, deviation from the user's typical transaction time pattern.
4. These features are passed into a trained ML model.
5. The model predicts whether the transaction is **Genuine** or **Suspicious/Fraudulent**.
6. Flagged transactions are highlighted/alerted on the dashboard.

---

## 📊 Model Details

- **Algorithm Used:** _(e.g., Logistic Regression / Random Forest / Decision Tree — update with your actual model)_
- **Key Features Used:** transaction time, time since last transaction, transaction amount, frequency of transactions, historical behavior pattern
- **Dataset:** _(mention if it's a public dataset, synthetic data, or custom-collected data)_

---

## 🔮 Future Improvements

- [ ] Add more fraud indicators (location-based, device-based, amount-pattern-based)
- [ ] Real-time streaming detection instead of batch checking
- [ ] Model retraining pipeline with new data
- [ ] Email/SMS alert integration for flagged transactions
- [ ] Admin dashboard with analytics and fraud trend charts
- [ ] Deploy as a live web service (Flask/Django + hosting)


---

⭐ If you found this project useful, consider giving it a star!
