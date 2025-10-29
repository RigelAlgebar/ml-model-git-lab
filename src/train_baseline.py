#!/usr/bin/env python3
"""
# note
Baseline RandomForest on iris_dummy.csv.

Run:
  python src/train_baseline.py \
      --input data/raw/iris_dummy.csv \
      --report reports/model_baseline.md \
      --model-path artifacts/model_rf.pkl
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/iris_dummy.csv", help="CSV with header, target column 'species'")
    p.add_argument("--test-size", type=float, default=0.2, help="Test fraction")
    p.add_argument("--n-estimators", type=int, default=50, help="RF trees")
    p.add_argument("--max-depth", type=int, default=None, help="RF max depth (None = unlimited)")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--report", default="reports/model_baseline.md")
    p.add_argument("--model-path", default="artifacts/model_rf.pkl")
    return p.parse_args()

def load_xy(path: str):
    df = pd.read_csv(path)
    y = df["species"]
    X = df.drop(columns=["species"])
    return X, y

def main():
    args = parse_args()
    os.makedirs("reports", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    X, y = load_xy(args.input)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    acc = accuracy_score(y_te, preds)
    report = classification_report(y_te, preds, zero_division=0)

    # Save model
    joblib.dump(clf, args.model_path)

    # Write markdown report
    with open(args.report, "w") as f:
        f.write("# Baseline RandomForest Report\n\n")
        f.write(f"- Input: `{args.input}`\n")
        f.write(f"- Test size: `{args.test_size}` | n_estimators: `{args.n_estimators}` | max_depth: `{args.max_depth}`\n")
        f.write(f"- **Accuracy**: `{acc:.3f}`\n\n")
        f.write("## Classification report\n\n")
        f.write("```\n")
        f.write(report)
        f.write("```\n")

    print(f"[OK] Saved model → {args.model_path}")
    print(f"[OK] Wrote report → {args.report}")
    print(f"[OK] Accuracy: {acc:.3f}")

if __name__ == "__main__":
    main()
