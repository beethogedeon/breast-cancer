# Breast Cancer Wisconsin — Diagnostic Classification

Binary classification (malignant / benign) using the
[Breast Cancer Wisconsin dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
(569 samples · 30 features · CC BY 4.0).

## Project structure

```
.
├── src/
│   ├── data_preprocessing.py   # load, clean, split, scale
│   ├── train.py                # XGBoost training + MLflow logging
│   └── evaluate.py             # confusion matrix, ROC, feature importance
├── tests/
│   └── test_data.py            # unit + integration tests (pytest)
├── requirements.txt
└── environment.yml
```

## Quick start

```bash
# 1. Create and activate environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Train (default hyper-params, best config found via grid search)
cd src
python train.py --max_depth 4 --n_estimators 200 --lr 0.05 \
                --alpha 0.1 --reg_lambda 1.0 --early_stopping 10

# 3. Evaluate & error analysis
python evaluate.py

# 4. Run tests
cd ..
pytest tests/ -v
```

## Model choice

| Model            | AUROC  | Note |
|------------------|--------|------|
| XGBoost          | ~0.997 | Best; handles non-linear interactions |
| Random Forest    | ~0.993 | Good, but slower inference |
| Logistic Reg.    | ~0.989 | Linear boundary; limited on this dataset |

Most impactful hyperparameter: **`max_depth`** (tuned in range 3–8; best = 4).

## Reproducibility

Seeds set in `data_preprocessing.py` and `train.py`:
- `random.seed(42)`
- `numpy.random.seed(42)`
- XGBoost `seed=42`

Remaining non-determinism: CUDA cuBLAS ops (not applicable here since CPU-only),
and OS-level thread scheduling in `n_jobs=-1` parallel trees.
