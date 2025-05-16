
"""
Module to load raw JSON and CSV files, optionally preprocess them, and convert them to optimized Parquet format.
Output is saved under data/processed/
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path("../data/raw")
PROCESSED_DIR = Path("../data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def convert_json_to_parquet(filename: str, output_name: str | None = None) -> None:
    """Load line-delimited JSON and save as Parquet."""
    json_path = RAW_DIR / filename
    df = pd.read_json(json_path, lines=True)
    if output_name is None:
        output_name = filename.replace(".json", ".parquet")
    df.to_parquet(PROCESSED_DIR / output_name, index=False)


def convert_csv_to_parquet(filename: str, output_name: str | None = None) -> None:
    """Load CSV file and save as Parquet."""
    csv_path = RAW_DIR / filename
    df = pd.read_csv(csv_path)
    if output_name is None:
        output_name = filename.replace(".csv", ".parquet")
    df.to_parquet(PROCESSED_DIR / output_name, index=False)


if __name__ == "__main__":
    convert_json_to_parquet("transactions.json")
    convert_csv_to_parquet("users.csv")
    convert_csv_to_parquet("merchants.csv")
    print("✅ All files converted to Parquet.")
