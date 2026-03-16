import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest
from data_preprocessing import load_data, clean_data, split_and_scale


@pytest.fixture(scope="module")
def raw_df():
    return load_data(save_csv=False)


@pytest.fixture(scope="module")
def clean_df(raw_df):
    return clean_data(raw_df.copy())


# Unit tests

class TestLoadData:
    def test_shape(self, raw_df):
        """Dataset must have 569 samples and 31 columns (30 features + label)."""
        assert raw_df.shape == (569, 31), f"Unexpected shape: {raw_df.shape}"

    def test_label_column_exists(self, raw_df):
        assert "label" in raw_df.columns

    def test_binary_labels(self, raw_df):
        assert set(raw_df["label"].unique()).issubset({0, 1})

    def test_no_missing_values(self, raw_df):
        assert raw_df.isnull().sum().sum() == 0


class TestCleanData:
    def test_no_duplicates(self, clean_df):
        assert clean_df.duplicated().sum() == 0

    def test_no_missing_after_clean(self, clean_df):
        assert clean_df.isnull().sum().sum() == 0

    def test_label_unchanged(self, raw_df, clean_df):
        """Cleaning must not alter the label column values."""
        assert set(clean_df["label"].unique()) == set(raw_df["label"].unique())

    def test_feature_count_preserved(self, raw_df, clean_df):
        assert clean_df.shape[1] == raw_df.shape[1]


# Integration test

class TestSplitAndScale:
    def test_split_sizes(self, clean_df):
        X_train, X_val, X_test, y_train, y_val, y_test, _ = split_and_scale(clean_df.copy())
        total = len(y_train) + len(y_val) + len(y_test)
        assert total == len(clean_df)
        # Each split must be non-empty
        assert len(y_train) > 0 and len(y_val) > 0 and len(y_test) > 0

    def test_train_scaler_zero_mean(self, clean_df):
        """After StandardScaler, train features should have ~0 mean."""
        X_train, _, _, _, _, _, _ = split_and_scale(clean_df.copy())
        np.testing.assert_allclose(X_train.mean(axis=0), 0, atol=1e-6)

    def test_stratification(self, clean_df):
        """Class balance should be roughly similar across splits."""
        _, _, _, y_train, y_val, y_test, _ = split_and_scale(clean_df.copy())
        train_ratio = y_train.mean()
        val_ratio   = y_val.mean()
        test_ratio  = y_test.mean()
        assert abs(train_ratio - val_ratio)  < 0.06
        assert abs(train_ratio - test_ratio) < 0.06

    def test_reproducibility(self, clean_df):
        """Two calls with same seed must produce identical splits."""
        splits_a = split_and_scale(clean_df.copy())
        splits_b = split_and_scale(clean_df.copy())
        np.testing.assert_array_equal(splits_a[3], splits_b[3])  # y_train
