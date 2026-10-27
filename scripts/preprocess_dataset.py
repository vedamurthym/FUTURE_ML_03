"""
Dataset Preprocessing Script for Stage 2.
Executes the full text preprocessing pipeline, flags empty and duplicate records,
saves the cleaned dataset to data/processed/resumes_cleaned.csv, and reports
aggregate statistics without exposing candidate names or raw resume contents.
"""

import os
import sys
from pathlib import Path

# Add project root directory to sys.path so 'src' is discoverable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data_loader import load_dataset
from src.text_preprocessor import (
    preprocess_dataframe,
    save_processed_data,
    get_modeling_dataset,
    DEFAULT_PROCESSED_PATH,
)


def run_preprocessing():
    print("=" * 70)
    print("STAGE 2: RESUME TEXT PROCESSING & PREPROCESSING PIPELINE")
    print("=" * 70)

    # 1. Load original dataset safely
    print("\n[Step 1] Loading raw dataset via src.data_loader...")
    try:
        raw_df = load_dataset()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    total_records = len(raw_df)
    print(f" -> Raw dataset loaded successfully: {total_records:,} rows, {raw_df.shape[1]} columns.")
    print(f" -> Input columns: {list(raw_df.columns)}")

    # 2. Run preprocessing pipeline
    print("\n[Step 2] Processing resume texts (normalization, tech terms, HTML fallback)...")
    processed_df = preprocess_dataframe(raw_df)

    # 3. Analyze usability
    usable_count = int(processed_df["is_usable"].sum())
    unusable_count = total_records - usable_count
    print(f"\n[Step 3] Usability Summary:")
    print(f" -> Total resumes evaluated: {total_records:,}")
    print(f" -> Usable resumes: {usable_count:,} ({(usable_count/total_records)*100:.2f}%)")
    print(f" -> Unusable/empty resumes: {unusable_count:,} ({(unusable_count/total_records)*100:.2f}%)")

    if unusable_count > 0:
        unusable_records = processed_df[~processed_df["is_usable"]]
        print("    Unusable record details:")
        for idx, row in unusable_records.iterrows():
            print(f"    - Index {idx}: ID={row['ID']}, Category={row['Category']}, Reason='Empty Resume_str and empty Resume_html'")

    # 4. Analyze duplicates
    dup_count = int(processed_df["is_duplicate"].sum())
    print(f"\n[Step 4] Deduplication Summary:")
    print(f" -> Duplicate texts flagged (secondary instances): {dup_count}")

    if dup_count > 0:
        dup_records = processed_df[processed_df["is_duplicate"]]
        print("    Flagged duplicate records:")
        for idx, row in dup_records.iterrows():
            # Find the primary record with same cleaned_text
            primary = processed_df[(processed_df["cleaned_text"] == row["cleaned_text"]) & (~processed_df["is_duplicate"])]
            primary_id = primary.iloc[0]["ID"] if len(primary) > 0 else "Unknown"
            print(f"    - Duplicate ID={row['ID']} (Category={row['Category']}) duplicates Primary ID={primary_id}")

    # 5. Usable & Unique Modeling Set
    modeling_df = get_modeling_dataset(processed_df)
    print(f"\n[Step 5] Training-Ready Clean Dataset:")
    print(f" -> Unique, usable resumes for ML: {len(modeling_df):,} records")

    # 6. Cleaned Text Length Statistics
    words_series = modeling_df["cleaned_text"].apply(lambda x: len(x.split()))
    chars_series = modeling_df["cleaned_text"].apply(len)
    print(f"\n[Step 6] Cleaned Text Statistics (Words & Characters):")
    print(f" -> Word count   : min={words_series.min()}, max={words_series.max():,}, mean={words_series.mean():.1f}, median={words_series.median():.1f}")
    print(f" -> Char count   : min={chars_series.min()}, max={chars_series.max():,}, mean={chars_series.mean():.1f}, median={chars_series.median():.1f}")

    # 7. Category Distribution
    print(f"\n[Step 7] Category Distribution (Top 10):")
    top_categories = modeling_df["Category"].value_counts().head(10)
    for cat, count in top_categories.items():
        print(f" -> {cat:26s}: {count:,} resumes")

    # 8. Save Processed Data safely
    print(f"\n[Step 8] Saving processed dataset...")
    output_path = save_processed_data(processed_df)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f" -> Saved to: {output_path}")
    print(f" -> File size: {file_size_mb:.2f} MB")
    print(f" -> Columns saved: {list(processed_df.columns)}")

    print("\n" + "=" * 70)
    print("STAGE 2 PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    run_preprocessing()
