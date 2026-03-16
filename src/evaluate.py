import os
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve
from sklearn.datasets import load_breast_cancer

from data_preprocessing import load_data, clean_data, split_and_scale

OUTPUT_DIR = "plots"


def load_results(checkpoint_dir: str = "checkpoints"):
    # load best model, all splits, and run results
    with open(os.path.join(checkpoint_dir, "results.json")) as f:
        results = json.load(f)

    best_name = results["_best"]
    best_model = joblib.load(os.path.join(checkpoint_dir, "best_model.pkl"))

    df = load_data(save_csv=False)
    df = clean_data(df)
    X_train, _, X_test, y_train, _, y_test, _ = split_and_scale(df)
    feature_names = load_breast_cancer(as_frame=True).feature_names

    return best_model, best_name, X_train, y_train, X_test, y_test, feature_names, results


def plot_model_comparison(results: dict, output_dir: str = OUTPUT_DIR):
    # grouped bar chart: train vs cv vs val AUROC for each model, with overfit gap annotation
    os.makedirs(output_dir, exist_ok=True)
    names = [k for k in results if not k.startswith("_")]
    names = sorted(names, key=lambda k: results[k]["cv_auroc"], reverse=True)

    train_aucs = [results[k]["train_auroc"] for k in names]
    cv_aucs    = [results[k]["cv_auroc"]    for k in names]
    val_aucs   = [results[k]["val_auroc"]   for k in names]
    gaps       = [results[k]["overfit_gap"] for k in names]

    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, train_aucs, w, label="Train AUROC", color="steelblue")
    ax.bar(x,     cv_aucs,    w, label="CV AUROC (5-fold)", color="seagreen")
    ax.bar(x + w, val_aucs,   w, label="Val AUROC", color="coral")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("AUROC"); ax.set_ylim(0.95, 1.01)
    ax.set_title("Model comparison — Train / CV / Val AUROC")
    ax.legend()
    for i, gap in enumerate(gaps):
        color = "red" if gap > 0.03 else "black"
        ax.text(i, max(train_aucs[i], val_aucs[i]) + 0.002,
                f"Δ={gap:+.3f}", ha="center", fontsize=8, color=color)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "model_comparison.png"), bbox_inches="tight")
    plt.close(fig)


def plot_learning_curve(model, X_train, y_train, model_name: str, output_dir: str = OUTPUT_DIR):
    # learning curve to detect overfitting: train vs cv score as training size increases
    os.makedirs(output_dir, exist_ok=True)
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        cv=5, scoring="roc_auc",
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1,
    )
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_sizes, train_mean, "o-", label="Train AUROC", color="steelblue")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="steelblue")
    ax.plot(train_sizes, val_mean, "o-", label="CV AUROC (5-fold)", color="coral")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="coral")
    ax.set_xlabel("Training set size"); ax.set_ylabel("AUROC")
    ax.set_title(f"Learning curve — {model_name}")
    ax.legend(); ax.set_ylim(0.88, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "learning_curve.png"), bbox_inches="tight")
    plt.close(fig)

    final_gap = round(float(train_mean[-1] - val_mean[-1]), 4)
    print(f"Learning curve — train={train_mean[-1]:.4f}  cv={val_mean[-1]:.4f}  gap={final_gap:+.4f}")
    if final_gap > 0.03:
        print("  WARNING: gap > 0.03 — possible overfitting")
    else:
        print("  OK: gap <= 0.03 — no significant overfitting")
    return final_gap


def plot_best_model(model, X_test, y_test, feature_names, output_dir: str = OUTPUT_DIR):
    # confusion matrix, ROC curve and feature importance for the selected model
    os.makedirs(output_dir, exist_ok=True)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["malignant", "benign"]).plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix — Test Set")
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), bbox_inches="tight")
    plt.close(fig)

    fn_mask = (y_test == 0) & (preds == 1)
    if fn_mask.sum() > 0:
        fn_samples = X_test[fn_mask]
        top_idx = np.argmax(np.abs(fn_samples.mean(axis=0)))
        print(f"FN={fn_mask.sum()}  mean_prob={probs[fn_mask].mean():.3f}  "
              f"dominant_feature='{feature_names[top_idx]}'")

    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend()
    fig.savefig(os.path.join(output_dir, "roc_curve.png"), bbox_inches="tight")
    plt.close(fig)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[-15:]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(np.array(feature_names)[top_idx], importances[top_idx])
        ax.set_title("Top 15 Feature Importances (gain)")
        fig.savefig(os.path.join(output_dir, "feature_importance.png"), bbox_inches="tight")
        plt.close(fig)

    return cm, auc


if __name__ == "__main__":
    model, best_name, X_train, y_train, X_test, y_test, feat_names, results = load_results()
    print(f"Best model: {best_name}")
    plot_model_comparison(results)
    plot_learning_curve(model, X_train, y_train, best_name)
    cm, auc = plot_best_model(model, X_test, y_test, feat_names)
    print(f"test auroc={auc:.4f}\n{cm}")
    print("Plots saved to plots/")
