"""
Candidate Ranking Demonstration Script for Stage 5.

Demonstrates candidate matching and ranking against a fictional Machine Learning
job description using the dual-method matching engine:
- Method A: TF-IDF vectorization and cosine similarity.
- Method B: Weighted skill matching (essential vs. optional coverage).
- Combined: Explainable 70/30 weighted scoring with deterministic tie-breaking.

Loads preprocessed dataset from data/processed/, preserves all records,
and displays top 10 candidates with scores and explanations.
Strictly uses fictional JD; displays NO candidate personal information.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.job_analyzer import analyze_job_description
from src.matching_engine import (
    load_processed_candidates_dataset,
    rank_candidates,
    DEFAULT_SCORING_CONFIG,
)

# ----------------------------------------------------------------------------
# FICTIONAL MACHINE LEARNING JOB DESCRIPTION
# ----------------------------------------------------------------------------
FICTIONAL_ML_JOB_DESCRIPTION = """
Job Title: Senior Machine Learning Engineer (Fictional Role)
Company: NexusAI Analytics Corp. (Fictional Company)
Department: AI Research & Production Engineering
Location: Remote / Flexible

About Us:
NexusAI Analytics builds intelligent predictive analytics platforms for
enterprise supply chain and financial intelligence.

Core Requirements:
- 4+ years of professional programming in Python and SQL.
- Strong practical experience with scikit-learn and Machine Learning modeling.
- Solid understanding of Linux environments and Docker containerization.
- Demonstrated problem-solving ability in production systems.

Preferred Qualifications:
- Experience deploying ML models on AWS or cloud infrastructure is preferred.
- Familiarity with PyTorch or Deep Learning frameworks is a plus.
- Hands-on experience with Apache Spark or Big Data pipelines is nice to have.

Professional Attributes:
- Excellent communication and cross-functional leadership skills.
"""


def main():
    print("=" * 78)
    print("STAGE 5: CANDIDATE RANKING & MATCHING ENGINE DEMONSTRATION")
    print("=" * 78)
    print("DISCLAIMER: The job description below is purely FICTIONAL.")
    print("Rankings are decision-support heuristics, NOT automated hiring decisions.")
    print("=" * 78)

    # 1. Load existing processed candidate dataset
    print("\n[Step 1] Loading processed resume dataset...")
    try:
        resumes_df = load_processed_candidates_dataset()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    total_records = len(resumes_df)
    print(f" -> Total records loaded: {total_records:,}")

    # 2. Analyze the fictional job description using the existing job analyzer
    print("\n[Step 2] Analyzing fictional Job Description via src.job_analyzer...")
    job_analysis = analyze_job_description(FICTIONAL_ML_JOB_DESCRIPTION)

    if not job_analysis["is_valid"]:
        print("Error: Job description failed validation.")
        sys.exit(1)

    print(f" -> Job Title          : Senior Machine Learning Engineer (Fictional)")
    print(f" -> Essential Skills ({len(job_analysis['essential_skills'])}): {', '.join(job_analysis['essential_skills'])}")
    print(f" -> Optional Skills  ({len(job_analysis['optional_skills'])}): {', '.join(job_analysis['optional_skills'])}")
    print(f" -> Unspecified Skills: {', '.join(job_analysis['unspecified_skills'])}")

    # 3. Execute Candidate Ranking
    print("\n[Step 3] Running Candidate Matching Engine (70% Skill / 30% Text Similarity)...")
    ranking_result = rank_candidates(
        job_analysis=job_analysis,
        resumes_df=resumes_df,
        top_n=10,
        scoring_config=DEFAULT_SCORING_CONFIG,
    )

    total_evaluated = ranking_result["total_evaluated"]
    top_candidates = ranking_result["ranked_candidates"]

    print(f" -> Total eligible candidates evaluated (non-duplicate, usable): {total_evaluated:,}")
    print(f" -> Scoring Weights: Skill Match = {DEFAULT_SCORING_CONFIG['weight_skill']*100:.0f}%, "
          f"Text Similarity = {DEFAULT_SCORING_CONFIG['weight_text']*100:.0f}%")
    print(f" -> Essential vs. Optional Skill Split: {DEFAULT_SCORING_CONFIG['essential_skill_weight']*100:.0f}% / "
          f"{DEFAULT_SCORING_CONFIG['optional_skill_weight']*100:.0f}%")

    # 4. Display Top 10 Candidates
    print("\n" + "=" * 78)
    print(f"{'TOP 10 RANKED CANDIDATES FOR FICTIONAL ML ROLE':^78}")
    print("=" * 78)

    for rank, cand in enumerate(top_candidates, start=1):
        cand_id = cand["candidate_id"]
        cat = cand["category"]
        comb_score = cand["combined_score"]
        skill_score = cand["skill_match_score"]
        text_sim = cand["text_similarity_score"]
        ess_matched = cand["essential_skills_matched"]
        ess_missing = cand["essential_skills_missing"]
        opt_matched = cand["optional_skills_matched"]
        ess_cov = cand["essential_coverage_percent"]

        print(f"\n#{rank:<2} [ID: {cand_id}] - Category: {cat}")
        print(f"    Scores     : Combined: {comb_score:.2f} | Skill Match: {skill_score:.2f} | Text Similarity: {text_sim:.2f}%")
        print(f"    Essential  : {len(ess_matched)}/{len(ess_matched)+len(ess_missing)} ({ess_cov:.1f}%) Matched: {', '.join(ess_matched) if ess_matched else 'None'}")
        if ess_missing:
            print(f"    Missing    : {', '.join(ess_missing)}")
        print(f"    Optional   : {', '.join(opt_matched) if opt_matched else 'None'}")
        print(f"    Explanation: \"{cand['match_explanation']}\"")

    print("\n" + "=" * 78)
    print("STAGE 5 CANDIDATE RANKING COMPLETED SUCCESSFULLY")
    print("=" * 78)


if __name__ == "__main__":
    main()
