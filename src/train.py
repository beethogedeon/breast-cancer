import os
import json
import random
import time
import joblib

import numpy as np
import mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, classification_report

from data_preprocessing import load_data, clean_data, split_and_scale
from models import MODEL_REGISTRY

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = "checkpoints"


def _evaluate_split(model, X, y):
    # returns auroc and f1 for a given split
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    return roc_auc_score(y, probs), f1_score(y, preds)


def run_grid_search(X_train, y_train, X_val, y_val):
    # grid search with 5-fold CV on train set, then val evaluation for each model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mlflow.set_experiment("breast-cancer-model-selection")

    results = {}

    for name, config in MODEL_REGISTRY.items():
        with mlflow.start_run(run_name=name):
            gs = GridSearchCV(
                config["estimator"],
                config["param_grid"],
                cv=5,
                scoring="roc_auc",
                n_jobs=-1,
                refit=True,
            )

            t0 = time.time()
            gs.fit(X_train, y_train)
            elapsed = time.time() - t0

            best = gs.best_estimator_
            train_auc, _        = _evaluate_split(best, X_train, y_train)
            val_auc,   val_f1   = _evaluate_split(best, X_val,   y_val)
            overfit_gap = round(train_auc - val_auc, 6)

            mlflow.log_params(gs.best_params_)
            mlflow.log_metrics({
                "cv_auroc":    gs.best_score_,
                "train_auroc": train_auc,
                "val_auroc":   val_auc,
                "val_f1":      val_f1,
                "overfit_gap": overfit_gap,
                "fit_time_s":  elapsed,
            })

            joblib.dump(best, os.path.join(OUTPUT_DIR, f"{name}.pkl"))

            results[name] = {
                "cv_auroc":    round(gs.best_score_, 6),
                "train_auroc": round(train_auc, 6),
                "val_auroc":   round(val_auc, 6),
                "val_f1":      round(val_f1, 6),
                "overfit_gap": overfit_gap,
                "best_params": gs.best_params_,
                "fit_time_s":  round(elapsed, 2),
            }

            flag = "  *** OVERFIT ***" if overfit_gap > 0.03 else ""
            print(f"  {name:25s}  train={train_auc:.4f}  cv={gs.best_score_:.4f}  val={val_auc:.4f}  gap={overfit_gap:+.4f}{flag}")

    return results


def select_and_evaluate(results, X_test, y_test):
    # load the best model by val AUROC and run final test evaluation
    best_name = max(results, key=lambda k: results[k]["val_auroc"])
    best_model = joblib.load(os.path.join(OUTPUT_DIR, f"{best_name}.pkl"))

    test_auc, test_f1 = _evaluate_split(best_model, X_test, y_test)
    test_preds = best_model.predict(X_test)

    print(f"\n  Best model : {best_name}")
    print(f"  test_auroc : {test_auc:.4f}")
    print(f"  test_f1    : {test_f1:.4f}")
    print(classification_report(y_test, test_preds, target_names=["malignant", "benign"]))

    results[best_name]["test_auroc"] = round(test_auc, 6)
    results[best_name]["test_f1"]    = round(test_f1, 6)
    results["_best"] = best_name

    joblib.dump(best_model, os.path.join(OUTPUT_DIR, "best_model.pkl"))
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return best_name, best_model, test_auc


def main():
    os.makedirs("data", exist_ok=True)
    df = load_data(save_csv=True)
    df = clean_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(df)
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

    print("\nGrid search across all models:")
    results = run_grid_search(X_train, y_train, X_val, y_val)

    print("\nComparison (sorted by cv AUROC):")
    print(f"  {'model':25s}  {'train':>6}  {'cv(5f)':>6}  {'val':>6}  {'gap':>7}  {'val_f1':>6}")
    for name, r in sorted(results.items(), key=lambda x: x[1]["cv_auroc"], reverse=True):
        flag = " ***" if r["overfit_gap"] > 0.03 else ""
        print(f"  {name:25s}  {r['train_auroc']:.4f}  {r['cv_auroc']:.4f}  {r['val_auroc']:.4f}  {r['overfit_gap']:+.4f}  {r['val_f1']:.4f}{flag}")

    print("\nFinal test evaluation:")
    select_and_evaluate(results, X_test, y_test)


if __name__ == "__main__":
    main()
