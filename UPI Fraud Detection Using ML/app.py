"""
UPI Fraud Detection — Flask Web Application
Serves the dashboard and prediction API.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib, json, os

app = Flask(__name__)

MODEL_DIR = "models"
DATA_PATH = "data/upi_transactions.csv"

# Load artifacts
def load_artifacts():
    artifacts = {}
    try:
        artifacts["model"] = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
        artifacts["scaler"] = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        artifacts["label_encoders"] = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
        artifacts["feature_cols"] = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
        with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
            artifacts["metrics"] = json.load(f)
        print("✅ All artifacts loaded successfully!")
    except Exception as e:
        print(f"⚠️  Error loading artifacts: {e}")
        print("   Run 'python train_model.py' first!")
    return artifacts

artifacts = load_artifacts()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/stats")
def api_stats():
    if "metrics" not in artifacts:
        return jsonify({"error": "Model not trained yet"}), 500
    return jsonify(artifacts["metrics"])

@app.route("/api/recent")
def api_recent():
    try:
        df = pd.read_csv(DATA_PATH)
        recent = df.tail(50).to_dict(orient="records")
        return jsonify(recent)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/lookup", methods=["POST"])
def api_lookup():
    """Lookup a transaction by Transaction ID or UPI ID from the dataset."""
    try:
        data = request.json
        query = data.get("query", "").strip()
        search_type = data.get("search_type", "transaction_id")  # transaction_id or upi_id

        if not query:
            return jsonify({"error": "Please enter a Transaction ID or UPI ID"}), 400

        df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

        if search_type == "transaction_id":
            # Exact match on transaction_id
            matches = df[df["transaction_id"].str.upper() == query.upper()]
        else:
            # Search by sender or receiver UPI ID
            matches = df[
                (df["sender_upi_id"].str.lower() == query.lower()) |
                (df["receiver_upi_id"].str.lower() == query.lower())
            ]

        if matches.empty:
            return jsonify({
                "found": False,
                "message": f"No transactions found for {'Transaction ID' if search_type == 'transaction_id' else 'UPI ID'}: {query}",
                "transactions": [],
                "summary": None
            })

        # Limit to 100 results for UPI ID searches
        display_matches = matches.head(100)

        # Build transaction list
        transactions = []
        for _, row in display_matches.iterrows():
            txn = {
                "transaction_id": row["transaction_id"],
                "timestamp": str(row["timestamp"]),
                "sender_upi_id": row["sender_upi_id"],
                "receiver_upi_id": row["receiver_upi_id"],
                "amount": round(float(row["amount"]), 2),
                "transaction_type": row["transaction_type"],
                "sender_bank": row["sender_bank"],
                "receiver_bank": row["receiver_bank"],
                "device_type": row["device_type"],
                "sender_city": row["sender_city"],
                "receiver_city": row["receiver_city"],
                "upi_app": row["upi_app"],
                "is_fraud": int(row["is_fraud"]),
                "sender_balance_before": round(float(row["sender_balance_before"]), 2),
                "sender_balance_after": round(float(row["sender_balance_after"]), 2),
            }
            transactions.append(txn)

        # Run ML prediction on matches
        model = artifacts.get("model")
        scaler = artifacts.get("scaler")
        le_dict = artifacts.get("label_encoders")
        feature_cols = artifacts.get("feature_cols")

        ml_predictions = []
        if model and scaler and le_dict and feature_cols:
            for _, row in display_matches.iterrows():
                ts = pd.Timestamp(row["timestamp"])
                hour = ts.hour
                day_of_week = ts.dayofweek
                amount = float(row["amount"])
                sender_bal = float(row["sender_balance_before"])
                receiver_bal = float(row["receiver_balance_before"])

                def safe_encode(le, val):
                    try:
                        return le.transform([str(val)])[0]
                    except ValueError:
                        return 0

                features = {
                    "amount": amount,
                    "log_amount": np.log1p(amount),
                    "sender_balance_before": sender_bal,
                    "sender_balance_after": float(row["sender_balance_after"]),
                    "receiver_balance_before": receiver_bal,
                    "receiver_balance_after": float(row["receiver_balance_after"]),
                    "hour": hour,
                    "day_of_week": day_of_week,
                    "is_weekend": 1 if day_of_week >= 5 else 0,
                    "is_night": 1 if (hour >= 22 or hour <= 5) else 0,
                    "month": ts.month,
                    "amount_to_balance_ratio": amount / (sender_bal + 1),
                    "balance_change_sender": sender_bal - float(row["sender_balance_after"]),
                    "balance_change_receiver": float(row["receiver_balance_after"]) - receiver_bal,
                    "balance_remaining_pct": float(row["sender_balance_after"]) / (sender_bal + 1),
                    "transaction_type_encoded": safe_encode(le_dict["transaction_type"], row["transaction_type"]),
                    "sender_bank_encoded": safe_encode(le_dict["sender_bank"], row["sender_bank"]),
                    "receiver_bank_encoded": safe_encode(le_dict["receiver_bank"], row["receiver_bank"]),
                    "device_type_encoded": safe_encode(le_dict["device_type"], row["device_type"]),
                    "sender_city_encoded": safe_encode(le_dict["sender_city"], row["sender_city"]),
                    "receiver_city_encoded": safe_encode(le_dict["receiver_city"], row["receiver_city"]),
                    "upi_app_encoded": safe_encode(le_dict["upi_app"], row["upi_app"]),
                }
                X = pd.DataFrame([features])[feature_cols]
                X.replace([np.inf, -np.inf], np.nan, inplace=True)
                X.fillna(0, inplace=True)
                X_scaled = scaler.transform(X)
                prob = float(model.predict_proba(X_scaled)[0][1])
                ml_predictions.append(round(prob * 100, 2))

        # Attach ML predictions to transactions
        for i, txn in enumerate(transactions):
            if i < len(ml_predictions):
                txn["ml_fraud_probability"] = ml_predictions[i]
            else:
                txn["ml_fraud_probability"] = None

        # Summary for UPI ID searches
        summary = {
            "total_transactions": len(matches),
            "fraud_count": int(matches["is_fraud"].sum()),
            "legit_count": int((matches["is_fraud"] == 0).sum()),
            "total_amount": round(float(matches["amount"].sum()), 2),
            "avg_amount": round(float(matches["amount"].mean()), 2),
            "fraud_rate": round(float(matches["is_fraud"].mean()) * 100, 2),
            "is_suspicious": bool(matches["is_fraud"].sum() > 0),
        }

        return jsonify({
            "found": True,
            "search_type": search_type,
            "query": query,
            "transactions": transactions,
            "summary": summary,
            "showing": len(transactions),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.json
        model = artifacts["model"]
        scaler = artifacts["scaler"]
        le_dict = artifacts["label_encoders"]
        feature_cols = artifacts["feature_cols"]

        amount = float(data.get("amount", 0))
        sender_bal = float(data.get("sender_balance", 10000))
        receiver_bal = float(data.get("receiver_balance", 5000))
        hour = int(data.get("hour", 12))
        txn_type = data.get("transaction_type", "P2P")
        device = data.get("device_type", "Android")
        sender_bank = data.get("sender_bank", "SBI")
        receiver_bank = data.get("receiver_bank", "HDFC")
        sender_city = data.get("sender_city", "Mumbai")
        receiver_city = data.get("receiver_city", "Delhi")
        upi_app = data.get("upi_app", "GPay")

        day_of_week = int(data.get("day_of_week", 2))
        is_weekend = 1 if day_of_week >= 5 else 0
        is_night = 1 if (hour >= 22 or hour <= 5) else 0
        month = int(data.get("month", 4))

        def safe_encode(le, val):
            try:
                return le.transform([val])[0]
            except ValueError:
                return 0

        features = {
            "amount": amount,
            "log_amount": np.log1p(amount),
            "sender_balance_before": sender_bal,
            "sender_balance_after": sender_bal - amount,
            "receiver_balance_before": receiver_bal,
            "receiver_balance_after": receiver_bal + amount,
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "is_night": is_night,
            "month": month,
            "amount_to_balance_ratio": amount / (sender_bal + 1),
            "balance_change_sender": amount,
            "balance_change_receiver": amount,
            "balance_remaining_pct": (sender_bal - amount) / (sender_bal + 1),
            "transaction_type_encoded": safe_encode(le_dict["transaction_type"], txn_type),
            "sender_bank_encoded": safe_encode(le_dict["sender_bank"], sender_bank),
            "receiver_bank_encoded": safe_encode(le_dict["receiver_bank"], receiver_bank),
            "device_type_encoded": safe_encode(le_dict["device_type"], device),
            "sender_city_encoded": safe_encode(le_dict["sender_city"], sender_city),
            "receiver_city_encoded": safe_encode(le_dict["receiver_city"], receiver_city),
            "upi_app_encoded": safe_encode(le_dict["upi_app"], upi_app),
        }

        X = pd.DataFrame([features])[feature_cols]
        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[0][1]
        pred = int(prob >= 0.5)

        risk = "Low" if prob < 0.3 else ("Medium" if prob < 0.6 else "High")

        return jsonify({
            "fraud_probability": round(float(prob) * 100, 2),
            "prediction": "Fraudulent" if pred else "Legitimate",
            "risk_level": risk,
            "is_fraud": pred
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🚀 UPI Fraud Detection Dashboard")
    print("  📍 http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000)
