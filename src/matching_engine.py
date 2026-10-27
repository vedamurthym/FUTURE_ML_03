"""
Candidate Ranking and Matching Engine.

Implements an explainable, dual-method candidate-to-job matching pipeline:
Method A: TF-IDF vectorization and cosine similarity over shared feature space.
Method B: Weighted skill matching (essential vs. optional coverage).
Combined: Transparent weighted score formula with deterministic tie-breaking.
"""

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.text_preprocessor import clean_text
from src.skill_extractor import extract_skills, infer_category

# Default paths for loading processed data
DEFAULT_PROCESSED_CSV = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "resumes_cleaned.csv"
)
DEFAULT_SKILLS_CSV = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "resumes_with_skills.csv"
)

# Default scoring weights (configurable)
DEFAULT_SCORING_CONFIG: Dict[str, float] = {
    "weight_skill": 0.70,           # Weight of skill match in combined score (70%)
    "weight_text": 0.30,            # Weight of TF-IDF similarity in combined score (30%)
    "essential_skill_weight": 0.80, # Weight of essential skills within skill score (when optional exist)
    "optional_skill_weight": 0.20,  # Weight of optional skills within skill score (when optional exist)
}

# Custom token pattern for TF-IDF that preserves discrete technical terms
TFIDF_TOKEN_PATTERN = (
    r"(?u)(?:(?<!\w)\.net\b|\bc\+\+(?!\+)|\bc\#(?!\w)|\bci\/cd\b|\bpl\/sql\b|\b\w+\b)"
)


# ============================================================================
# DATA LOADING & HELPER UTILITIES
# ============================================================================

def load_processed_candidates_dataset(
    cleaned_csv_path: Optional[Union[str, Path]] = None,
    skills_csv_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Load and join the processed resumes dataset with extracted skills.
    Preserves all records, usability flags, and duplicate flags.
    """
    clean_p = Path(cleaned_csv_path).resolve() if cleaned_csv_path else DEFAULT_PROCESSED_CSV
    skills_p = Path(skills_csv_path).resolve() if skills_csv_path else DEFAULT_SKILLS_CSV

    if not clean_p.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at: {clean_p}")

    df_clean = pd.read_csv(clean_p)

    if skills_p.exists():
        df_skills = pd.read_csv(skills_p)
        # Merge on ID, adding skills columns
        merge_cols = [c for c in ["skills", "skills_by_category", "skill_count"] if c in df_skills.columns]
        if merge_cols:
            df_merged = pd.merge(df_clean, df_skills[["ID"] + merge_cols], on="ID", how="left")
            return df_merged

    return df_clean


def parse_candidate_skills(val: Any) -> Set[str]:
    """
    Safely parse skills from candidate record (handles strings, lists, sets, NaN, None).
    """
    if val is None:
        return set()
    if isinstance(val, (list, set, tuple)):
        return {str(s).strip() for s in val if str(s).strip()}
    if isinstance(val, str):
        if not val.strip():
            return set()
        return {p.strip() for p in re.split(r"[;,]", val) if p.strip()}
    if isinstance(val, (float, int)) and (pd.isna(val) or math.isnan(val)):
        return set()
    return set()


# ============================================================================
# METHOD A: TF-IDF AND COSINE SIMILARITY
# ============================================================================

def compute_tfidf_similarities(
    cleaned_jd: str,
    candidate_texts: List[str],
) -> List[float]:
    """
    Compute TF-IDF cosine similarity between the cleaned job description
    and a list of candidate cleaned texts fitted in the exact same feature space.
    
    Returns:
        List of similarity scores scaled between 0.0 and 100.0.
    """
    if not cleaned_jd or not candidate_texts:
        return [0.0] * len(candidate_texts)

    # Combined corpus ensures shared vocabulary
    corpus = [cleaned_jd] + candidate_texts

    try:
        vectorizer = TfidfVectorizer(
            token_pattern=TFIDF_TOKEN_PATTERN,
            lowercase=True,
            stop_words=None,  # We retain clean normalized tokens without blind stopword stripping
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)
        jd_vector = tfidf_matrix[0:1]
        candidate_vectors = tfidf_matrix[1:]

        # Cosine similarity between JD and all candidates
        sims = cosine_similarity(jd_vector, candidate_vectors).flatten()
        # Scale to 0 - 100 and round to 2 decimals
        return [round(float(s) * 100.0, 2) for s in sims]

    except (ValueError, Exception):
        # Fallback if vocabulary is completely empty or all docs have no valid tokens
        return [0.0] * len(candidate_texts)


# ============================================================================
# METHOD B: WEIGHTED SKILL MATCHING
# ============================================================================

def compute_skill_metrics(
    candidate_skills: Set[str],
    essential_skills: Set[str],
    optional_skills: Set[str],
    scoring_config: Dict[str, float],
) -> Dict[str, Any]:
    """
    Compare candidate skills against job requirements and compute coverage
    and skill match scores with transparent fallbacks.
    """
    essential_matched = sorted(list(candidate_skills & essential_skills))
    essential_missing = sorted(list(essential_skills - candidate_skills))
    optional_matched = sorted(list(candidate_skills & optional_skills))
    optional_missing = sorted(list(optional_skills - candidate_skills))
    other_skills = sorted(list(candidate_skills - (essential_skills | optional_skills)))

    # Coverage calculations
    has_essential = len(essential_skills) > 0
    has_optional = len(optional_skills) > 0

    if has_essential:
        essential_cov = round((len(essential_matched) / len(essential_skills)) * 100.0, 2)
    else:
        essential_cov = None

    if has_optional:
        optional_cov = round((len(optional_matched) / len(optional_skills)) * 100.0, 2)
    else:
        optional_cov = None

    # Weights within the skill component
    w_ess = scoring_config.get("essential_skill_weight", 0.80)
    w_opt = scoring_config.get("optional_skill_weight", 0.20)

    # Calculate skill match score (0 - 100)
    if has_essential and has_optional:
        skill_score = round((essential_cov * w_ess) + (optional_cov * w_opt), 2)
    elif has_essential and not has_optional:
        skill_score = essential_cov
    elif not has_essential and has_optional:
        # Documented fallback when job has only optional skills
        skill_score = optional_cov
    else:
        # Documented fallback when job has no detected skill requirements
        skill_score = 0.0

    return {
        "essential_skills_matched": essential_matched,
        "essential_skills_missing": essential_missing,
        "optional_skills_matched": optional_matched,
        "optional_skills_missing": optional_missing,
        "other_skills_matched": other_skills,
        "essential_coverage_percent": essential_cov,
        "optional_coverage_percent": optional_cov,
        "skill_match_score": skill_score,
        "has_essential": has_essential,
        "has_optional": has_optional,
    }


# ============================================================================
# EXPLANATION GENERATOR
# ============================================================================

def generate_match_explanation(
    metrics: Dict[str, Any],
    text_similarity_score: float,
    skill_match_score: Optional[float] = None,
    combined_score: Optional[float] = None,
) -> str:
    """
    Generate an explainable, fact-based summary from actual matching values.
    """
    has_ess = metrics.get("has_essential", False)
    has_opt = metrics.get("has_optional", False)
    ess_matched = metrics.get("essential_skills_matched", [])
    ess_missing = metrics.get("essential_skills_missing", [])
    opt_matched = metrics.get("optional_skills_matched", [])
    opt_cov = metrics.get("optional_coverage_percent")
    s_score = skill_match_score if skill_match_score is not None else metrics.get("skill_match_score", 0.0)

    parts = []

    if has_ess:
        total_ess = len(ess_matched) + len(ess_missing)
        matched_cnt = len(ess_matched)
        if matched_cnt == total_ess:
            ess_str = f"Matched all {total_ess} essential skills ({', '.join(ess_matched[:4])})."
        elif matched_cnt == 0:
            ess_str = f"Matched 0 of {total_ess} essential skills. Missing {', '.join(ess_missing[:4])}."
        else:
            ess_str = (
                f"Matched {matched_cnt} of {total_ess} essential skills, including {', '.join(ess_matched[:3])}. "
                f"Missing {', '.join(ess_missing[:3])}."
            )
        parts.append(ess_str)
    elif has_opt:
        total_opt = len(opt_matched) + len(metrics.get("optional_skills_missing", []))
        parts.append(
            f"No essential requirements defined. Matched {len(opt_matched)} of {total_opt} optional skills ({opt_cov:.1f}%)."
        )
    else:
        parts.append("No specific skill requirements identified in the job description.")

    if has_opt and has_ess:
        if opt_matched:
            parts.append(f"Optional coverage: {opt_cov:.1f}% (including {', '.join(opt_matched[:2])}).")
        else:
            parts.append("Missing all optional skills.")

    parts.append(f"Skill score: {s_score:.1f}%.")
    parts.append(f"Text similarity was {text_similarity_score:.1f}%.")
    parts.append("Review the resume manually to verify proficiency.")

    return " ".join(parts)



def generate_detailed_explanation(
    metrics: Dict[str, Any],
    text_similarity_score: float,
    skill_match_score: Optional[float] = None,
    combined_score: Optional[float] = None,
) -> str:
    """
    Generate a comprehensive, evidence-based match explanation for the candidate profile page.
    """
    has_ess = metrics.get("has_essential", False)
    has_opt = metrics.get("has_optional", False)
    ess_matched = metrics.get("essential_skills_matched", [])
    ess_missing = metrics.get("essential_skills_missing", [])
    opt_matched = metrics.get("optional_skills_matched", [])
    opt_missing = metrics.get("optional_skills_missing", [])
    ess_cov = metrics.get("essential_coverage_percent")
    opt_cov = metrics.get("optional_coverage_percent")
    s_score = skill_match_score if skill_match_score is not None else metrics.get("skill_match_score", 0.0)
    c_score = combined_score if combined_score is not None else 0.0

    sections = []

    # 1. Essential Skills Analysis
    if has_ess:
        total_ess = len(ess_matched) + len(ess_missing)
        matched_cnt = len(ess_matched)
        if matched_cnt == total_ess:
            sections.append(f"Required Skills: Fully satisfied ({ess_cov:.1f}% coverage). Matched all {total_ess} required skills: {', '.join(ess_matched)}.")
        elif matched_cnt == 0:
            sections.append(f"Required Skills: 0.0% coverage. Missing all {total_ess} required skills: {', '.join(ess_missing)}.")
        else:
            sections.append(f"Required Skills: Partial coverage ({ess_cov:.1f}%). Matched {matched_cnt} of {total_ess} ({', '.join(ess_matched)}), but missing: {', '.join(ess_missing)}.")
    else:
        sections.append("Required Skills: No essential skills specified in the job description.")

    # 2. Optional Skills Analysis
    if has_opt:
        if opt_matched:
            sections.append(f"Optional Skills: {opt_cov:.1f}% coverage. Matched: {', '.join(opt_matched)}. Missing: {', '.join(opt_missing) if opt_missing else 'None'}.")
        else:
            sections.append(f"Optional Skills: No optional skills matched. Missing: {', '.join(opt_missing)}.")
    else:
        sections.append("Optional Skills: No optional skills specified in the job description.")

    # 3. Overall Scoring Assessment
    sections.append(
        f"Overall Match Assessment: Skill match score is {s_score:.1f}% and TF-IDF text similarity is {text_similarity_score:.1f}%, resulting in a combined score of {c_score:.1f}%. Evaluation is based on NLP keyword extraction and document text similarity. Manual recruiter review is advised to verify practical proficiency."
    )

    return " ".join(sections)


# ============================================================================
# CANDIDATE RANKING ENGINE
# ============================================================================

def rank_candidates(
    job_analysis: Dict[str, Any],
    resumes_df: pd.DataFrame,
    top_n: Optional[int] = None,
    scoring_config: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Rank eligible candidates against a structured job description analysis result.
    
    Args:
        job_analysis: Structured dictionary produced by src.job_analyzer.analyze_job_description.
        resumes_df: Processed resume dataset DataFrame.
        top_n: Optional number of top candidates to return.
        scoring_config: Optional dictionary overriding default scoring weights.
        
    Returns:
        Structured dictionary with status, candidate rankings, and validation messages.
    """
    # 1. Merge user scoring config with defaults
    cfg = dict(DEFAULT_SCORING_CONFIG)
    if scoring_config and isinstance(scoring_config, dict):
        cfg.update(scoring_config)

    w_skill = cfg.get("weight_skill", 0.70)
    w_text = cfg.get("weight_text", 0.30)

    # 2. Validate Job Analysis Input
    if not job_analysis or not isinstance(job_analysis, dict):
        return {
            "is_valid": False,
            "validation_messages": ["Job analysis input is missing or malformed."],
            "total_evaluated": 0,
            "scoring_config_used": cfg,
            "ranked_candidates": [],
        }

    if not job_analysis.get("is_valid", False):
        return {
            "is_valid": False,
            "validation_messages": job_analysis.get("validation_messages", ["Job description is invalid."]),
            "total_evaluated": 0,
            "scoring_config_used": cfg,
            "ranked_candidates": [],
        }

    essential_skills = set(job_analysis.get("essential_skills", []))
    optional_skills = set(job_analysis.get("optional_skills", []))
    cleaned_jd = job_analysis.get("cleaned_jd", "")

    validation_messages = list(job_analysis.get("validation_messages", []))

    # Warning if no skills in job description
    if not essential_skills and not optional_skills:
        validation_messages.append(
            "Notice: Job description contains no essential or optional skill requirements. "
            "Skill match score falls back to text similarity."
        )

    # 3. Filter Eligible Candidates (usable=True, is_duplicate=False)
    if resumes_df is None or len(resumes_df) == 0:
        return {
            "is_valid": False,
            "validation_messages": ["Resume dataset is empty."],
            "total_evaluated": 0,
            "scoring_config_used": cfg,
            "ranked_candidates": [],
        }

    # Usability filter
    mask = pd.Series(True, index=resumes_df.index)
    if "is_usable" in resumes_df.columns:
        mask &= resumes_df["is_usable"].astype(str).str.lower().isin(["true", "1"])
    if "is_duplicate" in resumes_df.columns:
        mask &= resumes_df["is_duplicate"].astype(str).str.lower().isin(["false", "0"])

    eligible_df = resumes_df[mask].copy()

    if len(eligible_df) == 0:
        return {
            "is_valid": True,
            "validation_messages": ["No eligible candidates available after usability and deduplication filtering."],
            "total_evaluated": 0,
            "scoring_config_used": cfg,
            "ranked_candidates": [],
        }

    # 4. Method A: Compute TF-IDF Text Similarities
    # Resolve candidate text column
    if "cleaned_text" in eligible_df.columns:
        cand_texts = list(eligible_df["cleaned_text"].fillna("").astype(str))
    elif "raw_text" in eligible_df.columns:
        cand_texts = [clean_text(t) for t in eligible_df["raw_text"]]
    else:
        cand_texts = [""] * len(eligible_df)

    text_similarities = compute_tfidf_similarities(cleaned_jd, cand_texts)

    # 5. Method B: Compute Weighted Skill Matches and Combined Scores
    ranked_records = []

    for i, (_, row) in enumerate(eligible_df.iterrows()):
        cand_id = row.get("ID") or row.get("candidate_id") or row.name

        # Resolve candidate skills
        cand_skills_raw = row.get("skills") or row.get("extracted_skills")
        if cand_skills_raw is not None and not (isinstance(cand_skills_raw, float) and pd.isna(cand_skills_raw)):
            cand_skills = parse_candidate_skills(cand_skills_raw)
        else:
            # Fallback extraction on the fly if skills column missing
            cand_skills = set(extract_skills(cand_texts[i]).get("skills", []))

        # Dynamic category inference if not present or placeholder
        category = row.get("Category") or row.get("category")
        if not category or str(category).strip().upper() in ("UNKNOWN", "UPLOADED"):
            category = infer_category(cand_skills, row.get("raw_text") or cand_texts[i])

        # Skill match metrics
        metrics = compute_skill_metrics(cand_skills, essential_skills, optional_skills, cfg)
        text_sim = text_similarities[i]
        skill_score = metrics["skill_match_score"]

        # If neither essential nor optional skills exist in JD, fallback to text similarity
        if not essential_skills and not optional_skills:
            skill_score = text_sim

        # Combined score calculation
        combined_score = round((w_skill * skill_score) + (w_text * text_sim), 2)

        explanation = generate_match_explanation(
            metrics, text_sim, skill_match_score=skill_score, combined_score=combined_score
        )
        detailed_explanation = generate_detailed_explanation(
            metrics, text_sim, skill_match_score=skill_score, combined_score=combined_score
        )

        ranked_records.append({
            "candidate_id": cand_id,
            "filename": row.get("filename") or str(cand_id),
            "category": category,
            "combined_score": combined_score,
            "skill_match_score": skill_score,
            "text_similarity_score": text_sim,
            "matched_essential": metrics["essential_skills_matched"],
            "missing_essential": metrics["essential_skills_missing"],
            "matched_optional": metrics["optional_skills_matched"],
            "missing_optional": metrics["optional_skills_missing"],
            "essential_skills_matched": metrics["essential_skills_matched"],
            "essential_skills_missing": metrics["essential_skills_missing"],
            "optional_skills_matched": metrics["optional_skills_matched"],
            "optional_skills_missing": metrics["optional_skills_missing"],
            "essential_coverage_percent": metrics["essential_coverage_percent"],
            "optional_coverage_percent": metrics["optional_coverage_percent"],
            "match_explanation": explanation,
            "detailed_explanation": detailed_explanation,
        })


    # 6. Deterministic Sorting
    # Primary: combined_score DESC
    # Tie-break 1: skill_match_score DESC
    # Tie-break 2: text_similarity_score DESC
    # Tie-break 3: candidate_id ASC (cast to string for universal comparison)
    ranked_records.sort(
        key=lambda r: (
            -r["combined_score"],
            -r["skill_match_score"],
            -r["text_similarity_score"],
            str(r["candidate_id"]),
        )
    )

    # Assign 1-based ranks
    for rank_idx, record in enumerate(ranked_records):
        record["candidate_rank"] = rank_idx + 1

    total_evaluated = len(ranked_records)

    # Slice top_n if requested
    if top_n is not None and top_n > 0:
        ranked_records = ranked_records[:top_n]

    return {
        "is_valid": True,
        "validation_messages": validation_messages,
        "total_evaluated": total_evaluated,
        "scoring_config_used": cfg,
        "essential_skills_count": len(essential_skills),
        "optional_skills_count": len(optional_skills),
        "ranked_candidates": ranked_records,
    }


def candidates_to_dataframe(ranking_result: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert the ranked candidates list from rank_candidates into a clean pandas DataFrame.
    """
    candidates = ranking_result.get("ranked_candidates", [])
    if not candidates:
        return pd.DataFrame()
    return pd.DataFrame(candidates)
