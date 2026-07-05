"""
UPI Fraud Detection — ML Training Pipeline
Trains Logistic Regression, Random Forest, and XGBoost models.
"""

import pandas as pd
import numpy as np
import os, json, joblib, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/upi_transactions.csv"
MODEL_DIR = "models"
PLOTS_DIR = "static/plots"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data():
    print("=" * 60)
    print("  UPI Fraud Detection — ML Training Pipeline")
    print("=" * 60)
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    print(f"\n📂 Loaded {len(df):,} rows | Fraud: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
    return df

def engineer_features(df):
    print("⚙️  Feature Engineering...")
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df["month"] = df["timestamp"].dt.month
    df["amount_to_balance_ratio"] = df["amount"] / (df["sender_balance_before"] + 1)
    df["balance_change_sender"] = df["sender_balance_before"] - df["sender_balance_after"]
    df["balance_change_receiver"] = df["receiver_balance_after"] - df["receiver_balance_before"]
    df["balance_remaining_pct"] = df["sender_balance_after"] / (df["sender_balance_before"] + 1)
    df["log_amount"] = np.log1p(df["amount"])

    label_encoders = {}
    cat_cols = ["transaction_type", "sender_bank", "receiver_bank",
                "device_type", "sender_city", "receiver_city", "upi_app"]
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    joblib.dump(label_encoders, os.path.join(MODEL_DIR, "label_encoders.pkl"))
    return df, label_encoders

def prepare_data(df):
    feature_cols = [
        "amount", "log_amount", "sender_balance_before", "sender_balance_after",
        "receiver_balance_before", "receiver_balance_after",
        "hour", "day_of_week", "is_weekend", "is_night", "month",
        "amount_to_balance_ratio", "balance_change_sender",
        "balance_change_receiver", "balance_remaining_pct",
        "transaction_type_encoded", "sender_bank_encoded", "receiver_bank_encoded",
        "device_type_encoded", "sender_city_encoded", "receiver_city_encoded",
        "upi_app_encoded"
    ]
    X = df[feature_cols].copy()
    y = df["is_fraud"].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    print(f"📋 Features: {X.shape[1]} | Samples: {X.shape[0]:,}")
    return X, y, feature_cols

def train_models(X, y, feature_cols):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    print("⚖️  Applying SMOTE...")
    sm = SMOTE(random_state=42, sampling_strategy=0.5)
    X_res, y_res = sm.fit_resample(X_train_s, y_train)

    models_dict = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, class_weight="balanced", n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
            scale_pos_weight=len(y_train[y_train==0])/max(len(y_train[y_train==1]),1),
            random_state=42, eval_metric="logloss", use_label_encoder=False)
    }

    results = {}
    best_auc, best_name = 0, ""
    for name, model in models_dict.items():
        print(f"\n  📌 Training {name}...")
        model.fit(X_res, y_res)
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        results[name] = {"accuracy": round(acc,4), "precision": round(prec,4), "recall": round(rec,4),
                         "f1_score": round(f1,4), "auc_roc": round(auc,4),
                         "confusion_matrix": cm.tolist(), "y_prob": y_prob.tolist()}
        print(f"     Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        if auc > best_auc:
            best_auc, best_name = auc, name
        joblib.dump(model, os.path.join(MODEL_DIR, name.lower().replace(" ", "_") + ".pkl"))

    print(f"\n  🏆 Best: {best_name} (AUC={best_auc:.4f})")
    joblib.dump(models_dict[best_name], os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_cols.pkl"))
    return results, y_test, best_name, X_test_s, models_dict

def generate_plots(df, results, y_test, models_dict, X_test_s, feature_cols):
    print("\n📊 Generating plots...")
    plt.style.use("dark_background")
    colors = ["#00d4ff", "#7c3aed", "#f43f5e", "#10b981", "#f59e0b"]
    bg = "#0f172a"

    # Fraud distribution pie
    fig, ax = plt.subplots(figsize=(6,6)); fig.patch.set_facecolor(bg)
    ax.pie(df["is_fraud"].value_counts().values, labels=["Legitimate","Fraudulent"],
           colors=["#10b981","#f43f5e"], autopct="%1.1f%%", startangle=90, explode=(0,0.08),
           textprops={"fontsize":14,"color":"white"})
    ax.set_title("Transaction Distribution", fontsize=16, color="white", pad=20)
    plt.savefig(f"{PLOTS_DIR}/fraud_distribution.png", dpi=150, bbox_inches="tight", facecolor=bg); plt.close()

    # Fraud by hour
    fig, ax = plt.subplots(figsize=(10,5)); fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    fh = df[df["is_fraud"]==1]["hour"].value_counts().reindex(range(24), fill_value=0)
    ax.bar(fh.index, fh.values, color=["#f43f5e" if h<6 else "#7c3aed" for h in range(24)])
    ax.set_xlabel("Hour", color="#94a3b8"); ax.set_ylabel("Fraud Count", color="#94a3b8")
    ax.set_title("Fraudulent Transactions by Hour", fontsize=16, color="white")
    ax.set_xticks(range(24)); ax.tick_params(colors="#94a3b8")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    plt.savefig(f"{PLOTS_DIR}/fraud_by_hour.png", dpi=150, bbox_inches="tight", facecolor=bg); plt.close()

    # Amount distribution
    fig, ax = plt.subplots(figsize=(10,5)); fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    ax.hist(df[df["is_fraud"]==0]["amount"].clip(upper=50000), bins=60, alpha=0.7, color="#10b981", label="Legit")
    ax.hist(df[df["is_fraud"]==1]["amount"].clip(upper=50000), bins=60, alpha=0.7, color="#f43f5e", label="Fraud")
    ax.set_xlabel("Amount (₹)", color="#94a3b8"); ax.set_ylabel("Count", color="#94a3b8")
    ax.set_title("Amount Distribution", fontsize=16, color="white"); ax.legend()
    ax.tick_params(colors="#94a3b8"); [ax.spines[s].set_visible(False) for s in ["top","right"]]
    plt.savefig(f"{PLOTS_DIR}/amount_distribution.png", dpi=150, bbox_inches="tight", facecolor=bg); plt.close()

    # Model comparison
    fig, ax = plt.subplots(figsize=(10,6)); fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    mnames = list(results.keys()); metrics = ["accuracy","precision","recall","f1_score","auc_roc"]
    x = np.arange(len(mnames)); w = 0.15
    for i, m in enumerate(metrics):
        ax.bar(x + i*w, [results[n][m] for n in mnames], w, label=m.replace("_"," ").title(), color=colors[i])
    ax.set_xticks(x + w*2); ax.set_xticklabels(mnames); ax.set_ylim(0,1.15)
    ax.set_title("Model Comparison", fontsize=16, color="white"); ax.legend(fontsize=9)
    ax.tick_params(colors="#94a3b8"); [ax.spines[s].set_visible(False) for s in ["top","right"]]
    plt.savefig(f"{PLOTS_DIR}/model_comparison.png", dpi=150, bbox_inches="tight", facecolor=bg); plt.close()

    # ROC curves
    fig, ax = plt.subplots(figsize=(8,8)); fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    for i, (name, res) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, color=colors[i], linewidth=2.5, label=f"{name} (AUC={res['auc_roc']:.3f})")
    ax.plot([0,1],[0,1],"w--",alpha=0.3)
    ax.set_xlabel("FPR", color="#94a3b8"); ax.set_ylabel("TPR", color="#94a3b8")
    ax.set_title("ROC Curves", fontsize=16, color="white"); ax.legend(fontsize=11)
    ax.tick_params(colors="#94a3b8")
    plt.savefig(f"{PLOTS_DIR}/roc_curves.png", dpi=150, bbox_inches="tight", facecolor=bg); plt.close()

    # Confusion matrices
    fig, axes = plt.subplots(1,3,figsize=(18,5)); fig.patch.set_facecolor(bg)
    for i, (name, res) in enumerate(results.items()):
        sns.heatmap(np.array(res["confusion_matrix"]), annot=True, fmt="d", cmap="RdPu", ax=axes[i],
                    xticklabels=["Legit","Fraud"], yticklabels=["Legit","Fraud"])
        axes[i].set_title(name, fontsize=14, color="white")
        axes[i].set_ylabel("Actual", color="#94a3b8"); axes[i].set_xlabel("Predicted", color="#94a3b8")
        axes[i].set_facecolor(bg)
    plt.savefig(f"{PLOTS_DIR}/confusion_matrices.png", dpi=150, bbox_inches="tight", facecolor=bg); plt.close()

    # Feature importance
    rf = models_dict["Random Forest"]
    imp = rf.feature_importances_; idx = np.argsort(imp)[::-1][:15]
    fig, ax = plt.subplots(figsize=(10,7)); fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    ax.barh(range(len(idx)), imp[idx][::-1], color=plt.cm.plasma(np.linspace(0.2,0.9,len(idx))))
    ax.set_yticks(range(len(idx))); ax.set_yticklabels([feature_cols[i] for i in idx][::-1])
    ax.set_title("Top 15 Feature Importance", fontsize=16, color="white")
    ax.tick_params(colors="#94a3b8"); [ax.spines[s].set_visible(False) for s in ["top","right"]]
    plt.savefig(f"{PLOTS_DIR}/feature_importance.png", dpi=150, bbox_inches="tight", facecolor=bg); plt.close()

    # Fraud by type
    fig, ax = plt.subplots(figsize=(8,5)); fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    fbt = df.groupby("transaction_type")["is_fraud"].mean()*100
    ax.bar(fbt.index, fbt.values, color=colors[:len(fbt)])
    ax.set_ylabel("Fraud Rate (%)", color="#94a3b8")
    ax.set_title("Fraud Rate by Type", fontsize=16, color="white")
    ax.tick_params(colors="#94a3b8"); [ax.spines[s].set_visible(False) for s in ["top","right"]]
    plt.savefig(f"{PLOTS_DIR}/fraud_by_type.png", dpi=150, bbox_inches="tight", facecolor=bg); plt.close()

    print("   ✅ All plots saved!")

def save_metrics(df, results, best_name):
    clean = {n: {k:v for k,v in r.items() if k != "y_prob"} for n, r in results.items()}
    metrics = {
        "dataset": {
            "total_transactions": len(df), "fraud_count": int(df["is_fraud"].sum()),
            "legit_count": int((df["is_fraud"]==0).sum()),
            "fraud_rate": round(df["is_fraud"].mean()*100, 2),
            "avg_amount": round(df["amount"].mean(), 2),
            "avg_fraud_amount": round(df[df["is_fraud"]==1]["amount"].mean(), 2),
            "avg_legit_amount": round(df[df["is_fraud"]==0]["amount"].mean(), 2),
        },
        "models": clean, "best_model": best_name,
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Metrics saved!")

def main():
    df = load_data()
    df, le = engineer_features(df)
    X, y, fc = prepare_data(df)
    results, y_test, best, X_test_s, mods = train_models(X, y, fc)
    generate_plots(df, results, y_test, mods, X_test_s, fc)
    save_metrics(df, results, best)
    print("\n✨ Done! Run 'python app.py' to launch dashboard.")

if __name__ == "__main__":
    main()
