import os
import sys
import pandas as pd

def inspect_dataset(csv_path):
    print("=" * 60)
    print("RESUME DATASET INSPECTION & VALIDATION")
    print("=" * 60)

    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        sys.exit(1)

    print(f"Dataset path: {csv_path}")
    file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        sys.exit(1)

    print("\n1. Dataset Dimensions:")
    print(f"   - Rows: {df.shape[0]:,}")
    print(f"   - Columns: {df.shape[1]}")
    print(f"   - Column names: {list(df.columns)}")

    print("\n2. Data Types & Missing Values:")
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        dtype = df[col].dtype
        print(f"   - {col}: dtype={dtype}, nulls={null_count:,} ({null_pct:.2f}%)")

    print("\n3. Duplicate Analysis:")
    exact_duplicates = df.duplicated().sum()
    print(f"   - Full-row exact duplicates: {exact_duplicates:,}")
    
    # Check duplicates on ID if ID exists
    if "ID" in df.columns:
        id_duplicates = df["ID"].duplicated().sum()
        print(f"   - Duplicate IDs: {id_duplicates:,}")

    # Check duplicates on text if Resume_str or similar exists
    text_cols = [c for c in df.columns if "resume" in c.lower() or "text" in c.lower()]
    for tc in text_cols:
        text_dups = df[tc].duplicated().sum()
        print(f"   - Duplicate values in '{tc}': {text_dups:,} ({(text_dups/len(df))*100:.2f}%)")

    print("\n4. Category Column Analysis:")
    cat_cols = [c for c in df.columns if "category" in c.lower() or "label" in c.lower() or "role" in c.lower()]
    if cat_cols:
        for cc in cat_cols:
            unique_cats = df[cc].nunique()
            print(f"   - Column '{cc}' has {unique_cats} unique categories.")
            print("   - Top categories distribution:")
            top_cats = df[cc].value_counts().head(10)
            for cat, count in top_cats.items():
                print(f"       * {cat}: {count:,}")
            if unique_cats > 10:
                print(f"       ... and {unique_cats - 10} more categories.")
    else:
        print("   - No explicit category column detected automatically.")

    print("\n5. Text Characteristics (Without Exposing PII):")
    for tc in text_cols:
        lengths = df[tc].dropna().astype(str).apply(len)
        words = df[tc].dropna().astype(str).apply(lambda x: len(x.split()))
        print(f"   - Column '{tc}':")
        print(f"       * Character length: min={lengths.min()}, max={lengths.max():,}, mean={lengths.mean():.1f}, median={lengths.median():.1f}")
        print(f"       * Word count: min={words.min()}, max={words.max():,}, mean={words.mean():.1f}, median={words.median():.1f}")
        print(f"       * Empty strings / whitespace-only: {(df[tc].astype(str).str.strip() == '').sum():,}")

    print("\n6. Sample Record Schema Preview (PII-safe metadata only):")
    for i, row in df.head(3).iterrows():
        print(f"   - Record #{i}:")
        for col in df.columns:
            val = str(row[col])
            if col in text_cols:
                # Obfuscate / summarize length instead of printing raw text
                preview = f"<{len(val)} characters, {len(val.split())} words, starts with: '{val[:40].strip()}...'>"
                print(f"       {col}: {preview}")
            else:
                print(f"       {col}: {val}")

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE - DATASET LOADED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    csv_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "archive", "Resume", "Resume.csv")
    inspect_dataset(csv_file)
