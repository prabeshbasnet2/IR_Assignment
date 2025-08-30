# train_classifier.py
import os, glob, json, argparse, random
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import joblib

# ---------------- Config ----------------
DATA_DIR = Path("../crawler/data/classification")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "news_clf.joblib"
SUMMARY_PATH = MODEL_DIR / "news_clf_summary.json"
CATEGORIES = ["politics", "business", "health"]
RANDOM_STATE = 42

# ---------------- Data ----------------
def load_dataset():
    texts, labels = [], []
    for cat in CATEGORIES:
        folder = DATA_DIR / cat
        if not folder.exists():
            continue
        for fp in glob.glob(str(folder / "*.txt")):
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read().strip()
                if txt:
                    texts.append(txt)
                    labels.append(cat)
            except Exception as e:
                print(f"[warn] failed reading {fp}: {e}")
    return texts, labels

# ---------------- Pipelines ----------------
def build_pipeline(algo: str):
    vec = TfidfVectorizer(stop_words="english", max_df=0.8, ngram_range=(1, 2))

    if algo == "nb":
        clf = MultinomialNB(alpha=0.3)
    elif algo == "lr":
        # ✅ Fixed Logistic Regression config for Windows/Python 3.13
        clf = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            n_jobs=1,       # force single-thread to avoid joblib crash
            C=2.0,
            random_state=RANDOM_STATE
        )
    elif algo == "svm":
        clf = LinearSVC(C=1.0, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown algo: {algo}")

    return Pipeline([("tfidf", vec), ("clf", clf)])

# ---------------- Training & Eval ----------------
def evaluate_cv(pipe, X, y, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    cv = cross_validate(
        pipe, X, y, cv=skf,
        scoring=["f1_macro", "accuracy"],
        n_jobs=1,  # safe single-thread CV
        return_train_score=False
    )
    return {
        "cv_f1_macro_mean": float(np.mean(cv["test_f1_macro"])),
        "cv_acc_mean": float(np.mean(cv["test_accuracy"]))
    }

def heldout_report(pipe, X_te, y_te):
    y_hat = pipe.predict(X_te)
    return {
        "test_accuracy": accuracy_score(y_te, y_hat),
        "test_f1_macro": f1_score(y_te, y_hat, average="macro"),
        "classification_report": classification_report(y_te, y_hat, labels=CATEGORIES, zero_division=0, digits=4),
        "confusion_matrix": confusion_matrix(y_te, y_hat, labels=CATEGORIES).tolist()
    }

def train_and_eval(algo, X_tr, y_tr, X_te, y_te, folds=5):
    pipe = build_pipeline(algo)

    print(f"\nTraining {algo.upper()}...")
    cv_stats = evaluate_cv(pipe, X_tr, y_tr, folds=folds)
    print(f"[CV] {algo.upper()} F1-macro={cv_stats['cv_f1_macro_mean']:.4f}, ACC={cv_stats['cv_acc_mean']:.4f}")

    pipe.fit(X_tr, y_tr)
    held = heldout_report(pipe, X_te, y_te)
    print("[Held-out] Accuracy:", held["test_accuracy"])
    print("[Held-out] Macro-F1:", held["test_f1_macro"])
    print(held["classification_report"])
    return pipe, cv_stats, held

def main(test_size=0.2, folds=5, model="all"):
    random.seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)

    X, y = load_dataset()
    n = len(X)
    assert n >= 100, f"Need at least 100 docs; found {n}"
    print(f"[data] Loaded {n} documents")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)

    to_run = [model] if model in {"nb","lr","svm"} else ["nb","lr","svm"]

    results, best = {}, {"name": None, "pipe": None, "f1": -1}
    for algo in to_run:
        pipe, cv_stats, held = train_and_eval(algo, X_tr, y_tr, X_te, y_te, folds)
        f1m = held["test_f1_macro"]
        results[algo] = {"cv": cv_stats, "heldout": {"f1_macro": f1m, "accuracy": held["test_accuracy"]}}
        if f1m > best["f1"]:
            best.update({"name": algo, "pipe": pipe, "f1": f1m})

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"pipeline": best["pipe"], "labels": CATEGORIES}
    joblib.dump(payload, MODEL_PATH)
    print(f"\n[saved] {MODEL_PATH.resolve()}")

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "labels": CATEGORIES,
        "best_model": best["name"],
        "per_model": results
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {SUMMARY_PATH.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--model", type=str, choices=["nb","lr","svm","all"], default="all")
    args = parser.parse_args()
    main(test_size=args.test_size, folds=args.folds, model=args.model)
