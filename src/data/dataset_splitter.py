# ./src/data/dataset_splitter.py

import numpy as np
import pandas as pd
from typing import Tuple


def split_corpus_by_usfm(
        df: pd.DataFrame,
        group_col: str = "usfm_code",
        train_ratio: float = 0.90,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
        seed: int = 1989
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a parallel corpus into Train, Validation, and Test sets by grouping
    unique USFM verse codes to strictly prevent bible cross-edition Data Leakage.

    Args:
        df: Clean aligned DataFrame.
        group_col: Column name containing verse identifiers (e.g. 'usfm_code').
        train_ratio: Proportion of unique verse keys for training.
        val_ratio: Proportion of unique verse keys for validation.
        test_ratio: Proportion of unique verse keys for testing.
        seed: Random seed for reproducible splitting.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df).
    """

    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"

    # Extract unique verse codes
    unique_usfm_codes = np.sort(df[group_col].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_usfm_codes)

    n_total = len(unique_usfm_codes)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_keys = set(unique_usfm_codes[:n_train])
    val_keys = set(unique_usfm_codes[n_train:n_train + n_val])
    test_keys = set(unique_usfm_codes[n_train + n_val:])

    # Map rows to respective splits
    train_df = df[df[group_col].isin(train_keys)].reset_index(drop=True)
    val_df = df[df[group_col].isin(val_keys)].reset_index(drop=True)
    test_df = df[df[group_col].isin(test_keys)].reset_index(drop=True)

    print("\n" + "=" * 60)
    print("BIBLE DATASET SPLIT SUMMARY")
    print("=" * 60)
    print(f"Unique USFM Verses : {n_total:,}")
    print(f" -> Train Verses   : {len(train_keys):,} ({len(train_df):,} parallel pairs)")
    print(f" -> Val Verses     : {len(val_keys):,} ({len(val_df):,} parallel pairs)")
    print(f" -> Test Verses    : {len(test_keys):,} ({len(test_df):,} parallel pairs)")
    print("=" * 60 + "\n")

    return train_df, val_df, test_df


def split_dialogue_corpus(
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 1989
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a parallel dialogue corpus into Train (Adaptation) and Test (Out-of-Domain) sets.

    Args:
        df: Clean dialogue DataFrame.
        train_ratio: Proportion of pairs for adaptation/train.
        val_ratio: Proportion of pairs for validation.
        test_ratio: Proportion of pairs for testing.
        seed: Random seed for reproducible splitting.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df).
    """

    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"

    n_total = len(df)
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(n_total)
    df_shuffled = df.iloc[shuffled_indices].reset_index(drop=True)

    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_df = df_shuffled.iloc[:n_train].reset_index(drop=True)
    val_df = df_shuffled.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = df_shuffled.iloc[n_train + n_val:].reset_index(drop=True)

    print("\n" + "=" * 60)
    print("DIALOGUE (BOOK) DATASET SPLIT SUMMARY")
    print("=" * 60)
    print(f"Total Dialogue Pairs        : {n_total:,}")
    print(f" -> Train Pairs (OOD): {len(train_df):,}")
    print(f" -> Val Pairs (OOD)  : {len(val_df):,}")
    print(f" -> Test Pairs (OOD)  : {len(test_df):,}")
    print("=" * 60 + "\n")

    return train_df, val_df, test_df