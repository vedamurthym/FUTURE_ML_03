"""
Skill Extraction Execution Script for Stage 3.
Processes cleaned resumes from data/processed/resumes_cleaned.csv,
computes aggregate skill statistics on unique, usable resumes,
prints category distributions and PII-safe sample extractions,
and saves the enriched dataset to data/processed/resumes_with_skills.csv.
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path
import pandas as pd

# Add project root to sys.path so 'src' is discoverable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.skill_extractor import (
    DEFAULT_CLEANED_DATA_PATH,
    DEFAULT_SKILLS_DATA_PATH,
    extract_skills,
    process_resumes_skills,
    save_skills_dataset,
    DISCLAIMER_TEXT,
    SKILL_DEFINITIONS,
)


def run_skill_extraction():
    print("=" * 70)
    print("STAGE 3: NLP-BASED SKILL EXTRACTION PIPELINE")
    print("=" * 70)

    # 1. Load preprocessed dataset
    print(f"\n[Step 1] Loading preprocessed data from {DEFAULT_CLEANED_DATA_PATH.name}...")
    if not DEFAULT_CLEANED_DATA_PATH.exists():
        print(f"Error: {DEFAULT_CLEANED_DATA_PATH} not found. Please run scripts/preprocess_dataset.py first.")
        sys.exit(1)

    cleaned_df = pd.read_csv(DEFAULT_CLEANED_DATA_PATH)
    total_records = len(cleaned_df)
    print(f" -> Total records loaded: {total_records:,}")

    # 2. Filter for unique, usable evaluation subset
    usable_unique_df = cleaned_df[cleaned_df["is_usable"] & (~cleaned_df["is_duplicate"])].copy()
    eval_count = len(usable_unique_df)
    print(f"\n[Step 2] Evaluation Subset:")
    print(f" -> Unique, usable resumes for aggregate analysis: {eval_count:,} records")

    # 3. Execute extraction on all records (preserving full dataset structure)
    print("\n[Step 3] Extracting skills using explainable regex rule dictionary...")
    skills_df = process_resumes_skills(cleaned_df)

    # Filter processed skills for unique, usable subset for honest evaluation
    eval_skills_df = skills_df[skills_df["is_usable"] & (~skills_df["is_duplicate"])].copy()

    # 4. Compute aggregate skill metrics
    total_skill_mentions = int(eval_skills_df["total_skill_mentions"].sum())
    skill_counts = eval_skills_df["skill_count"]
    
    print(f"\n[Step 4] Aggregate Extraction Metrics:")
    print(f" -> Total skill mentions across dataset: {total_skill_mentions:,}")
    print(f" -> Average unique skills per candidate: {skill_counts.mean():.1f}")
    print(f" -> Median unique skills per candidate : {skill_counts.median():.1f}")
    print(f" -> Min / Max skills per candidate    : {skill_counts.min()} / {skill_counts.max()}")
    print(f" -> Candidates with at least 1 skill  : {(skill_counts > 0).sum():,} ({((skill_counts > 0).sum() / eval_count) * 100:.1f}%)")

    # 5. Calculate skill frequencies
    all_skills_flat = []
    for s_str in eval_skills_df["skills"]:
        if s_str:
            all_skills_flat.extend([s.strip() for s in s_str.split(";") if s.strip()])

    skill_freq = Counter(all_skills_flat)
    
    print("\n[Step 5] Top 20 Most Frequently Extracted Skills:")
    print(f" {'Rank':<5} {'Skill Name':<28} {'Mentions':<10} {'% of Resumes':<12}")
    print("-" * 58)
    for rank, (skill, count) in enumerate(skill_freq.most_common(20), 1):
        pct = (count / eval_count) * 100
        print(f" {rank:<5} {skill:<28} {count:<10,} {pct:.1f}%")

    # 6. Category-level analysis
    # Map each skill to category
    skill_to_cat = {s["name"]: s["category"] for s in SKILL_DEFINITIONS}
    cat_skills: dict = {}
    for s, cat in skill_to_cat.items():
        if cat not in cat_skills:
            cat_skills[cat] = Counter()
        if s in skill_freq:
            cat_skills[cat][s] = skill_freq[s]

    print("\n[Step 6] Top Skills by Category:")
    for cat, counter in cat_skills.items():
        top_in_cat = counter.most_common(5)
        top_str = ", ".join([f"{s} ({c:,})" for s, c in top_in_cat]) if top_in_cat else "None"
        print(f" -> {cat:<24}: {top_str}")

    # 7. Safe Sample Extraction Previews (Strictly No PII)
    print("\n[Step 7] Sample Extraction Previews (PII-Safe Metadata & Snippets):")
    sample_rows = eval_skills_df[eval_skills_df["skill_count"] >= 3].head(3)
    
    for idx, row in sample_rows.iterrows():
        sample_id = row["ID"]
        category = row["Category"]
        matched_skills = row["skills"].split("; ")
        
        # Get raw text for this candidate to show snippet
        raw_row = cleaned_df[cleaned_df["ID"] == sample_id].iloc[0]
        extraction = extract_skills(raw_row["raw_text"], window_size=35)
        
        print(f"\n  Candidate ID       : {sample_id} [Category: {category}]")
        print(f"  Extracted Skills ({len(matched_skills)}): {', '.join(matched_skills[:8])}{'...' if len(matched_skills) > 8 else ''}")
        
        # Show 2 representative evidence snippets
        print("  Evidence Snippets  :")
        shown = 0
        for skill_name, evidence_list in extraction["evidence"].items():
            if shown >= 2:
                break
            if evidence_list:
                ev = evidence_list[0]
                print(f"    * [{skill_name}] \"{ev['snippet']}\"")
                shown += 1

    # 8. Save Processed Dataset with Skills
    print(f"\n[Step 8] Saving enriched dataset with skills...")
    saved_path = save_skills_dataset(skills_df)
    file_size_mb = os.path.getsize(saved_path) / (1024 * 1024)
    print(f" -> Output file: {saved_path}")
    print(f" -> File size  : {file_size_mb:.2f} MB")
    print(f" -> Total rows : {len(skills_df):,} (preserving all raw records with flags)")
    print(f" -> Disclaimer : \"{DISCLAIMER_TEXT}\"")

    print("\n" + "=" * 70)
    print("STAGE 3 SKILL EXTRACTION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    run_skill_extraction()
