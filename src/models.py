from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# registry of candidate models and their hyperparameter grids for GridSearchCV
MODEL_REGISTRY = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=5000, random_state=42, solver="saga"),
        "param_grid": {
            "C": [0.01, 0.1, 1, 10],
            "l1_ratio": [0.0, 0.5, 1.0],  # 0=L2, 1=L1, in-between=ElasticNet
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, None],
            "min_samples_split": [2, 5],
        },
    },
    "svm": {
        "estimator": SVC(probability=True, random_state=42),
        "param_grid": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        },
    },
    "gradient_boosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [3, 4],
            "learning_rate": [0.05, 0.1],
        },
    },
    "xgboost": {
        "estimator": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            seed=42,
            n_jobs=-1,
            verbosity=0,
        ),
        "param_grid": {
            "max_depth": [3, 4, 5],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [100, 200],
            "reg_alpha": [0, 0.1],
            "reg_lambda": [1.0, 2.0],
        },
    },
}
