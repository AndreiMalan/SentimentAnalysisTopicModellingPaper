"""
Data Loading Module
===================
Load and validate Comments_and_brands.xlsx and Topics.xlsx datasets.
"""

import pandas as pd
import os


def load_comments(path: str = "Comments_and_brands.xlsx") -> pd.DataFrame:
    """Load the YouTube comments dataset.

    Expected columns: Comment, Brand
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Comments file not found at '{path}'. "
            "Place Comments_and_brands.xlsx in the project root."
        )

    df = pd.read_excel(path)

    expected = {"Comment", "Brand"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(
            f"Expected columns {expected}, got {list(df.columns)}. "
            "Ensure the Excel has 'Comment' and 'Brand' columns."
        )

    df = df.dropna(subset=["Comment"])
    df["Comment"] = df["Comment"].astype(str)
    df["Brand"] = df["Brand"].astype(str).str.strip()

    return df


def load_topics(path: str = "Topics.xlsx") -> pd.DataFrame:
    """Load the literature constructs / topics dataset.

    Expected columns: Construct, Inclusion criteria
    Optional: Literature, Theoretical definition, Exclusion criteria
    """
    if not os.path.exists(path):
        return None

    df = pd.read_excel(path)
    return df


def dataset_summary(df: pd.DataFrame) -> dict:
    """Return basic statistics about the comments dataset."""
    return {
        "total_comments": len(df),
        "unique_brands": df["Brand"].nunique(),
        "brands": df["Brand"].value_counts().to_dict(),
        "avg_comment_length": df["Comment"].str.len().mean(),
        "median_comment_length": df["Comment"].str.len().median(),
        "null_comments": df["Comment"].isna().sum(),
    }
