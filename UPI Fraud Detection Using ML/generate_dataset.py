"""
UPI Fraud Detection — Synthetic Dataset Generator
Generates ~100,000 realistic UPI transactions with injected fraud patterns.
"""

import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

# ─── Configuration ──────────────────────────────────────────────────────────────
NUM_TRANSACTIONS = 100_000
FRAUD_RATE = 0.025  # ~2.5% fraud
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "upi_transactions.csv")

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

# ─── Reference Data ─────────────────────────────────────────────────────────────
BANKS = ["SBI", "HDFC", "ICICI", "Axis", "Kotak", "PNB", "BOB", "Canara", "IndusInd", "Yes Bank"]
CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
    "Chandigarh", "Bhopal", "Patna", "Kochi", "Indore"
]
TRANSACTION_TYPES = ["P2P", "P2M", "Bill Payment", "Recharge"]
DEVICE_TYPES = ["Android", "iOS", "Web"]
UPI_APPS = ["GPay", "PhonePe", "Paytm", "BHIM", "Amazon Pay", "WhatsApp Pay"]

# ─── Helper Functions ────────────────────────────────────────────────────────────

def generate_upi_id(bank):
    """Generate a realistic UPI ID."""
    names = [
        "rahul", "priya", "amit", "sneha", "vijay", "anita", "suresh", "kavita",
        "rajesh", "meena", "deepak", "pooja", "arun", "divya", "mohan", "neha",
        "sanjay", "ritu", "vikram", "swati", "ashok", "geeta", "manoj", "sunita",
        "rakesh", "nisha", "kiran", "anjali", "prakash", "seema"
    ]
    suffixes = ["@oksbi", "@okhdfcbank", "@okicici", "@okaxis", "@ybl", "@paytm", "@ibl", "@upi"]
    name = random.choice(names) + str(random.randint(10, 9999))
    suffix = random.choice(suffixes)
    return f"{name}{suffix}"


def generate_timestamp(start_date, end_date):
    """Generate a random timestamp between two dates."""
    delta = end_date - start_date
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start_date + timedelta(seconds=random_seconds)


# ─── Main Dataset Generation ────────────────────────────────────────────────────

def generate_dataset():
    print("=" * 60)
    print("  UPI Fraud Detection — Dataset Generator")
    print("=" * 60)

    # Date range: past 6 months
    end_date = datetime(2026, 4, 1)
    start_date = end_date - timedelta(days=180)

    # Pre-generate user pool
    num_users = 5000
    user_pool = []
    for _ in range(num_users):
        bank = random.choice(BANKS)
        user_pool.append({
            "upi_id": generate_upi_id(bank),
            "bank": bank,
            "city": random.choice(CITIES),
            "avg_balance": np.random.lognormal(mean=10, sigma=1.2),  # Avg balance ₹5K-₹500K
            "preferred_device": random.choice(DEVICE_TYPES),
            "preferred_app": random.choice(UPI_APPS),
        })

    num_fraud = int(NUM_TRANSACTIONS * FRAUD_RATE)
    num_legit = NUM_TRANSACTIONS - num_fraud

    print(f"\n📊 Generating {NUM_TRANSACTIONS:,} transactions...")
    print(f"   ├── Legitimate: {num_legit:,}")
    print(f"   └── Fraudulent: {num_fraud:,} ({FRAUD_RATE*100:.1f}%)\n")

    records = []

    # ── Generate Legitimate Transactions ──
    print("  ✅ Generating legitimate transactions...")
    for i in range(num_legit):
        sender = random.choice(user_pool)
        receiver = random.choice(user_pool)
        while receiver["upi_id"] == sender["upi_id"]:
            receiver = random.choice(user_pool)

        timestamp = generate_timestamp(start_date, end_date)
        txn_type = np.random.choice(
            TRANSACTION_TYPES,
            p=[0.40, 0.35, 0.15, 0.10]
        )

        # Normal transaction amounts
        if txn_type == "Recharge":
            amount = round(random.choice([49, 99, 149, 199, 249, 299, 399, 499, 599, 699, 799]), 2)
        elif txn_type == "Bill Payment":
            amount = round(np.random.lognormal(mean=6.5, sigma=0.8), 2)
            amount = min(amount, 25000)
        elif txn_type == "P2M":
            amount = round(np.random.lognormal(mean=5.5, sigma=1.0), 2)
            amount = min(amount, 50000)
        else:  # P2P
            amount = round(np.random.lognormal(mean=6.0, sigma=1.2), 2)
            amount = min(amount, 100000)

        sender_balance = round(sender["avg_balance"] * np.random.uniform(0.5, 2.0), 2)
        sender_balance = max(sender_balance, amount + 100)  # Ensure sufficient balance

        records.append({
            "transaction_id": f"UPI{i+1:08d}",
            "timestamp": timestamp,
            "sender_upi_id": sender["upi_id"],
            "receiver_upi_id": receiver["upi_id"],
            "amount": amount,
            "transaction_type": txn_type,
            "sender_bank": sender["bank"],
            "receiver_bank": receiver["bank"],
            "sender_balance_before": sender_balance,
            "sender_balance_after": round(sender_balance - amount, 2),
            "receiver_balance_before": round(receiver["avg_balance"] * np.random.uniform(0.3, 1.8), 2),
            "receiver_balance_after": round(receiver["avg_balance"] * np.random.uniform(0.3, 1.8) + amount, 2),
            "device_type": sender["preferred_device"],
            "sender_city": sender["city"],
            "receiver_city": receiver["city"],
            "upi_app": sender["preferred_app"],
            "is_fraud": 0
        })

    # ── Generate Fraudulent Transactions ──
    print("  🚨 Injecting fraud patterns...")

    fraud_patterns = [
        "large_night",       # Large amount during odd hours
        "rapid_burst",       # Multiple transactions in quick succession
        "balance_drain",     # Draining account to near zero
        "unusual_amount",    # Amount way above user's normal
        "cross_city",        # Unusual city change
    ]

    for i in range(num_fraud):
        idx = num_legit + i
        sender = random.choice(user_pool)
        receiver = random.choice(user_pool)
        while receiver["upi_id"] == sender["upi_id"]:
            receiver = random.choice(user_pool)

        pattern = random.choice(fraud_patterns)

        if pattern == "large_night":
            # Large transactions between midnight and 4 AM
            hour = random.randint(0, 4)
            base_date = generate_timestamp(start_date, end_date)
            timestamp = base_date.replace(hour=hour, minute=random.randint(0, 59))
            amount = round(np.random.uniform(15000, 95000), 2)

        elif pattern == "rapid_burst":
            # Transactions happening within minutes
            timestamp = generate_timestamp(start_date, end_date)
            amount = round(np.random.uniform(5000, 49000), 2)

        elif pattern == "balance_drain":
            # Drain nearly the entire balance
            timestamp = generate_timestamp(start_date, end_date)
            sender_balance_temp = round(sender["avg_balance"] * np.random.uniform(0.8, 1.5), 2)
            amount = round(sender_balance_temp * np.random.uniform(0.85, 0.98), 2)

        elif pattern == "unusual_amount":
            # Amount 10x-50x the user's average
            timestamp = generate_timestamp(start_date, end_date)
            amount = round(sender["avg_balance"] * np.random.uniform(0.3, 0.8), 2)
            amount = min(amount, 99000)
            amount = max(amount, 5000)

        else:  # cross_city
            timestamp = generate_timestamp(start_date, end_date)
            amount = round(np.random.uniform(8000, 60000), 2)

        txn_type = np.random.choice(TRANSACTION_TYPES, p=[0.55, 0.30, 0.10, 0.05])
        sender_balance = round(sender["avg_balance"] * np.random.uniform(0.5, 2.0), 2)
        sender_balance = max(sender_balance, amount + 10)

        # Fraudsters often use different devices
        device = random.choice(DEVICE_TYPES)

        # Fraud sender city differs from registered city
        fraud_city = random.choice([c for c in CITIES if c != sender["city"]])

        records.append({
            "transaction_id": f"UPI{idx+1:08d}",
            "timestamp": timestamp,
            "sender_upi_id": sender["upi_id"],
            "receiver_upi_id": receiver["upi_id"],
            "amount": amount,
            "transaction_type": txn_type,
            "sender_bank": sender["bank"],
            "receiver_bank": receiver["bank"],
            "sender_balance_before": sender_balance,
            "sender_balance_after": round(sender_balance - amount, 2),
            "receiver_balance_before": round(receiver["avg_balance"] * np.random.uniform(0.1, 0.5), 2),
            "receiver_balance_after": round(receiver["avg_balance"] * np.random.uniform(0.1, 0.5) + amount, 2),
            "device_type": device,
            "sender_city": fraud_city if pattern == "cross_city" else sender["city"],
            "receiver_city": receiver["city"],
            "upi_app": random.choice(UPI_APPS),
            "is_fraud": 1
        })

    # ── Shuffle and save ──
    print("  🔀 Shuffling dataset...")
    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Re-assign sequential IDs after shuffle
    df["transaction_id"] = [f"UPI{i+1:08d}" for i in range(len(df))]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n  💾 Dataset saved to: {OUTPUT_FILE}")
    print(f"     Shape: {df.shape}")
    print(f"\n  📈 Summary Statistics:")
    print(f"     Amount  — Mean: ₹{df['amount'].mean():,.2f}  |  Median: ₹{df['amount'].median():,.2f}")
    print(f"     Amount  — Min:  ₹{df['amount'].min():,.2f}   |  Max: ₹{df['amount'].max():,.2f}")
    print(f"     Fraud   — {df['is_fraud'].sum():,} / {len(df):,} ({df['is_fraud'].mean()*100:.2f}%)")
    print(f"\n  ✨ Dataset generation complete!")
    print("=" * 60)

    return df


if __name__ == "__main__":
    generate_dataset()
