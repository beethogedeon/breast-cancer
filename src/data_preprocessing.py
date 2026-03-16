import random
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_data(save_csv: bool = True, csv_path: str = "data/breast_cancer.csv") -> pd.DataFrame:
    # load dataset and optionally persist it as CSV
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.rename(columns={"target": "label"})
    if save_csv:
        df.to_csv(csv_path, index=False)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # drop duplicates and clip extreme values beyond 3×IQR per feature
    df = df.drop_duplicates()

    feature_cols = [c for c in df.columns if c != "label"]
    for col in feature_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(lower=Q1 - 3 * IQR, upper=Q3 + 3 * IQR)

    return df


def split_and_scale(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15):
    # stratified 70/15/15 split, scaler fitted on train only to avoid leakage
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size / (1 - test_size),
        stratify=y_temp, random_state=RANDOM_SEED
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    split_and_scale(df)
