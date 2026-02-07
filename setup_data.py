"""
Setup Data Script
=================
Creates Topics.xlsx from the literature construct definitions in config/constructs.py.
Also creates a small sample Comments_and_brands.xlsx for testing.

Run once:
    python setup_data.py
"""

import pandas as pd
from config.constructs import CONSTRUCTS


def create_topics_excel(path: str = "Topics.xlsx"):
    """Create Topics.xlsx from construct definitions."""
    rows = []
    for name, info in CONSTRUCTS.items():
        rows.append({
            "Construct": name,
            "Literature": info["literature"],
            "Theoretical definition": info["definition"],
            "Inclusion criteria": ", ".join(info["inclusion_keywords"]),
            "Exclusion criteria": ", ".join(info.get("exclusion_keywords", [])),
        })
    df = pd.DataFrame(rows)
    df.to_excel(path, index=False)
    print(f"Created {path} with {len(df)} constructs.")
    return df


def create_sample_comments(path: str = "Comments_and_brands_sample.xlsx"):
    """Create a small sample dataset for testing the pipeline."""
    comments = [
        ("I love the camera quality on this phone, truly premium build!", "Samsung"),
        ("Not trustworthy at all, feels like a scam product", "Xiaomi"),
        ("This video is so funny and entertaining, love it!", "Huawei"),
        ("Very useful features, practical for everyday use", "Samsung"),
        ("Too expensive for what you get, not worth the price", "Xiaomi"),
        ("I recommend this to everyone, best brand for the money", "Samsung"),
        ("Great battery life and fast charging, impressive specs", "Huawei"),
        ("Don't trust this brand, quality has gone down", "Xiaomi"),
        ("Amazing display and screen quality, awesome design", "Samsung"),
        ("Good value for the price, affordable and budget friendly", "Huawei"),
        ("Should buy this phone, totally worth buying!", "Samsung"),
        ("The processor is efficient, great software updates", "Xiaomi"),
        ("Fun to watch, incredible content, catchy videos", "Huawei"),
        ("Weak build quality, not durable at all", "Xiaomi"),
        ("This is impractical, don't use if you need convenience", "Samsung"),
        ("Cheap but reliable, good for the price", "Huawei"),
        ("Avoid this product, worst I have ever used", "Xiaomi"),
        ("Innovation at its best, long-lasting performance", "Samsung"),
        ("Worth it, works well, very convenient upgrade", "Huawei"),
        ("Regret buying this, stop buying from this brand", "Xiaomi"),
    ]
    df = pd.DataFrame(comments, columns=["Comment", "Brand"])
    df.to_excel(path, index=False)
    print(f"Created {path} with {len(df)} sample comments.")
    return df


if __name__ == "__main__":
    create_topics_excel()
    create_sample_comments()
    print("\nSetup complete. Place your actual Comments_and_brands.xlsx in the project root.")
