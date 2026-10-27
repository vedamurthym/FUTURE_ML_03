"""
FastAPI Backend API for Standalone Resume Screening and Candidate Ranking.

Completely upload-based:
- Evaluates only uploaded resumes against the provided job description.
- Never loads, falls back to, or merges with any external/Kaggle datasets.
- Reuses existing core NLP & ML modules:
  - src.document_parser.parse_single_resume, parse_job_description_file
  - src.job_analyzer.analyze_job_description
  - src.matching_engine.rank_candidates, DEFAULT_SCORING_CONFIG
  - src.skill_extractor.SKILL_TAXONOMY
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure project root is discoverable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_parser import (
    parse_job_description_file,
    parse_multiple_resumes,
    parse_single_resume,
)
from src.job_analyzer import analyze_job_description
from src.matching_engine import (
    DEFAULT_SCORING_CONFIG,
    parse_candidate_skills,
    rank_candidates,
)
from src.skill_extractor import SKILL_TAXONOMY, infer_category

# ============================================================================
# APP INITIALIZATION & CORS CONFIGURATION
# ============================================================================

app = FastAPI(
    title="Resume Screening & Candidate Ranking API",
    description="Standalone upload-based NLP & ML candidate screening system",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def restore_vercel_rewritten_path(request: Request, call_next):
    """
    Handle Vercel internal rewrites:
    If a request was rewritten internally (e.g. to /api/index.py or /index.py),
    restore the original requested path from Vercel's x-forwarded-uri or x-matched-path.
    """
    original_uri = request.headers.get("x-forwarded-uri") or request.headers.get("x-matched-path")
    if original_uri:
        path_only = original_uri.split("?")[0]
        if request.url.path in ("/api/index.py", "/api/index", "/index.py", "/api/"):
            request.scope["path"] = path_only
    return await call_next(request)


# APIRouter allows endpoints to be served both with and without the /api prefix,
# guaranteeing routing success regardless of how Vercel rewrites internal paths.
api_router = APIRouter()

# In-memory candidate store for the current active screening session
# Maps candidate_id (filename or string ID) -> candidate metadata dict
CURRENT_SCREENING_CANDIDATES: Dict[str, Dict[str, Any]] = {}



def parse_skills_field(val: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
    """Helper to parse skill inputs passed as JSON arrays or comma-delimited strings."""
    if val is None:
        return None
    if isinstance(val, list):
        return [str(s).strip() for s in val if str(s).strip()]
    if isinstance(val, str):
        v = val.strip()
        if not v:
            return None
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return [str(s).strip() for s in parsed if str(s).strip()]
        except Exception:
            pass
        return [p.strip() for p in v.split(",") if p.strip()]
    return None


# ============================================================================
# PYDANTIC REQUEST & RESPONSE MODELS
# ============================================================================

class AnalyzeJobRequest(BaseModel):
    job_description: str = Field(..., description="Raw job description text")
    recruiter_config: Optional[Dict[str, List[str]]] = Field(
        default=None, description="Optional essential and optional skill overrides"
    )


class RankRequest(BaseModel):
    job_description: str = Field(..., description="Job description text")
    essential_skills: Optional[List[str]] = Field(
        default=None, description="Confirmed essential skills"
    )
    optional_skills: Optional[List[str]] = Field(
        default=None, description="Confirmed optional skills"
    )
    category_filter: Optional[str] = Field(
        default="ALL", description="Filter candidates by category (or 'ALL')"
    )
    min_score: Optional[float] = Field(
        default=0.0, ge=0.0, le=100.0, description="Minimum combined score threshold"
    )
    top_n: Optional[int] = Field(
        default=None, ge=1, description="Maximum number of candidates to return"
    )
    scoring_config: Optional[Dict[str, float]] = Field(
        default=None, description="Custom scoring weights"
    )


# ============================================================================
# API ENDPOINTS
# ============================================================================

@api_router.get("/health")
def health_check():
    """Health check endpoint returning system status and current upload session counts."""
    return {
        "status": "healthy",
        "service": "resume_screening_api",
        "mode": "standalone_upload",
        "current_candidates_count": len(CURRENT_SCREENING_CANDIDATES),
        "default_scoring_weights": DEFAULT_SCORING_CONFIG,
    }


@api_router.get("/categories")
def get_categories():
    """Return categories derived from active screening candidates, or taxonomy defaults."""
    if CURRENT_SCREENING_CANDIDATES:
        cat_counts: Dict[str, int] = {}
        for cand in CURRENT_SCREENING_CANDIDATES.values():
            c = str(cand.get("category", "Other / Unclassified"))
            cat_counts[c] = cat_counts.get(c, 0) + 1
        categories_list = [{"name": cat, "count": cnt} for cat, cnt in sorted(cat_counts.items())]
        return {
            "categories": categories_list,
            "total_categories": len(categories_list),
            "source": "session",
        }
    categories_list = [
        {"name": cat, "count": len(skills)}
        for cat, skills in sorted(SKILL_TAXONOMY.items())
    ]
    return {
        "categories": categories_list,
        "total_categories": len(categories_list),
        "source": "taxonomy",
    }


@api_router.post("/analyze-job")
def analyze_job(request: AnalyzeJobRequest):
    """
    Analyze a job description text, extract skills, and identify
    essential, optional, and unspecified requirements.
    """
    if not request.job_description or not request.job_description.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job description cannot be empty or whitespace-only.",
        )

    analysis = analyze_job_description(
        jd_text=request.job_description,
        recruiter_config=request.recruiter_config,
    )

    if not analysis.get("is_valid", False):
        msg = analysis.get("validation_messages", ["Invalid job description."])[0]
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)

    return analysis


@api_router.post("/upload-resumes")
async def upload_resumes(files: List[UploadFile] = File(...)):
    """
    Receive, validate, and parse multiple candidate resume documents (PDF & DOCX).
    Stores parsed results in the current session without touching any external dataset.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files uploaded. Please upload at least one PDF or DOCX file.",
        )

    file_items = []
    for f in files:
        try:
            content = await f.read()
            file_items.append((f.filename or "unnamed_resume", content))
        except Exception:
            file_items.append((f.filename or "unnamed_resume", b""))

    parsed_result = parse_multiple_resumes(file_items)

    # Cache valid records for step-2 interactive ranking
    global CURRENT_SCREENING_CANDIDATES
    CURRENT_SCREENING_CANDIDATES.clear()
    for item in parsed_result.get("files", []):
        if item.get("is_valid"):
            fname = item["filename"]
            inferred_cat = infer_category(
                item.get("extracted_skills", []), item.get("raw_text", "")
            )
            item["category"] = inferred_cat
            CURRENT_SCREENING_CANDIDATES[fname] = {
                "candidate_id": fname,
                "filename": fname,
                "category": inferred_cat,
                "raw_text": item.get("raw_text", ""),
                "cleaned_text": item.get("cleaned_text", ""),
                "skills": item.get("extracted_skills", []),
                "skill_count": item.get("skill_count", 0),
                "is_usable": True,
                "is_duplicate": False,
                "word_count": item.get("word_count", 0),
                "text_preview": item.get("text_preview", ""),
                "file_size_formatted": item.get("file_size_formatted", ""),
                "file_type": item.get("file_type", ""),
            }

    return parsed_result



@api_router.post("/screen")
async def screen_resumes_endpoint(
    job_description: Optional[str] = Form(None),
    jd_file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    resumes: Optional[List[UploadFile]] = File(None),
    essential_skills: Optional[str] = Form(None),
    optional_skills: Optional[str] = Form(None),
    category_filter: Optional[str] = Form("ALL"),
    min_score: Optional[float] = Form(0.0),
    top_n: Optional[int] = Form(None),
):
    """
    Primary standalone screening endpoint:
    1. Reads job description from pasted text or uploaded TXT/PDF file.
    2. Parses all uploaded PDF/DOCX resumes in-memory.
    3. Excludes any unreadable/invalid files with clear error reporting.
    4. Ranks only the valid uploaded candidates against the job description.
    5. Completely isolates from external/Kaggle datasets.
    """
    # 1. Resolve and validate Job Description
    jd_text = ""
    if jd_file is not None and jd_file.filename:
        try:
            content = await jd_file.read()
            jd_text = parse_job_description_file(jd_file.filename, content)
        except ValueError as ve:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job description error in '{jd_file.filename}': {str(ve)}",
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to read job description file '{jd_file.filename}': {str(e)}",
            )
    elif job_description and job_description.strip():
        jd_text = job_description.strip()

    if not jd_text or not jd_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No job description provided. Please paste text or upload a TXT or PDF file.",
        )

    # 2. Resolve uploaded resume files
    uploaded_files = files or resumes or []
    if not uploaded_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No resumes uploaded. Please upload at least one candidate resume (PDF or DOCX).",
        )

    # 3. Parse uploaded resumes directly in-memory
    valid_records = []
    failed_files = []

    for f in uploaded_files:
        clean_name = f.filename or "unnamed_resume"
        try:
            content = await f.read()
            parsed = parse_single_resume(clean_name, content)
        except Exception as e:
            parsed = {
                "filename": clean_name,
                "is_valid": False,
                "error": f"Failed to read resume file: {str(e)}",
                "extracted_skills": [],
                "raw_text": "",
                "cleaned_text": "",
                "word_count": 0,
                "text_preview": "",
                "file_size_formatted": "0 B",
                "file_type": "UNKNOWN",
            }

        if parsed["is_valid"]:
            valid_records.append(parsed)
        else:
            failed_files.append({
                "filename": parsed["filename"],
                "error": parsed["error"] or "Unknown file parsing error.",
            })

    if not valid_records:
        error_reasons = "; ".join([f"{f['filename']}: {f['error']}" for f in failed_files])
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"None of the uploaded resumes could be parsed ({len(failed_files)} file(s) failed). Details: {error_reasons}",
        )

    # 4. Analyze Job Description
    recruiter_config = None
    ess_list = parse_skills_field(essential_skills)
    opt_list = parse_skills_field(optional_skills)
    if ess_list is not None or opt_list is not None:
        recruiter_config = {
            "essential_skills": ess_list or [],
            "optional_skills": opt_list or [],
        }

    job_analysis = analyze_job_description(
        jd_text=jd_text,
        recruiter_config=recruiter_config,
    )

    if not job_analysis.get("is_valid", False):
        msg = job_analysis.get("validation_messages", ["Invalid job description."])[0]
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)

    # 5. Build in-memory candidate DataFrame exclusively from valid uploaded resumes
    df_rows = []
    for item in valid_records:
        inferred_cat = infer_category(item["extracted_skills"], item.get("raw_text", ""))
        df_rows.append({
            "ID": item["filename"],
            "candidate_id": item["filename"],
            "filename": item["filename"],
            "Category": inferred_cat,
            "category": inferred_cat,
            "raw_text": item["raw_text"],
            "cleaned_text": item["cleaned_text"],
            "skills": item["extracted_skills"],
            "extracted_skills": item["extracted_skills"],
            "is_usable": True,
            "is_duplicate": False,
            "word_count": item["word_count"],
            "text_preview": item["text_preview"],
            "file_size_formatted": item.get("file_size_formatted", ""),
            "file_type": item.get("file_type", "FILE"),
        })

    uploaded_df = pd.DataFrame(df_rows)

    # 6. Rank uploaded candidates using existing matching engine
    ranking_result = rank_candidates(
        job_analysis=job_analysis,
        resumes_df=uploaded_df,
        top_n=None,
        scoring_config=None,
    )

    all_ranked_candidates = ranking_result.get("ranked_candidates", [])

    # Enrich metadata for all candidates before session caching
    for cand in all_ranked_candidates:
        fname = cand.get("filename") or str(cand.get("candidate_id", ""))
        meta = next((r for r in valid_records if r["filename"] == fname), None)
        if meta:
            cand["raw_text"] = meta.get("raw_text", "")
            cand["cleaned_text"] = meta.get("cleaned_text", "")
            cand["skills"] = meta.get("extracted_skills", [])
            cand["extracted_skills"] = meta.get("extracted_skills", [])
            cand["all_skills"] = meta["extracted_skills"]
            cand["skill_count"] = len(meta["extracted_skills"])
            cand["word_count"] = meta["word_count"]
            cand["text_preview"] = meta["text_preview"]
            cand["file_size_formatted"] = meta["file_size_formatted"]
            cand["file_type"] = meta.get("file_type", "FILE")
        if not cand.get("category") or str(cand["category"]).upper() in ("UNKNOWN", "UPLOADED"):
            cand["category"] = infer_category(cand.get("all_skills", []), cand.get("text_preview", ""))

    # 7. Update current session store: replace previous candidates with the current request
    global CURRENT_SCREENING_CANDIDATES
    CURRENT_SCREENING_CANDIDATES.clear()
    for cand in all_ranked_candidates:
        cid = str(cand["candidate_id"])
        CURRENT_SCREENING_CANDIDATES[cid] = dict(cand)

    available_categories = sorted(list({c.get("category", "Other / Unclassified") for c in all_ranked_candidates}))

    # Filter by category if requested and not ALL
    filtered = all_ranked_candidates
    if category_filter and category_filter.strip().upper() not in ("ALL", "ALL CATEGORIES", ""):
        filtered = [c for c in filtered if c.get("category", "").strip().lower() == category_filter.strip().lower()]

    # Filter by min_score if requested
    if min_score and min_score > 0.0:
        filtered = [c for c in filtered if c["combined_score"] >= min_score]

    # Slice top_n if requested
    returned_candidates = filtered[:top_n] if top_n and top_n > 0 else filtered

    # Re-assign 1-based ranks
    for idx, cand in enumerate(returned_candidates):
        cand["candidate_rank"] = idx + 1

    # Merge validation messages
    all_validation_msgs = list(ranking_result.get("validation_messages", []))
    for m in job_analysis.get("validation_messages", []):
        if m not in all_validation_msgs:
            all_validation_msgs.append(m)

    return {
        "is_valid": True,
        "total_uploaded": len(uploaded_files),
        "total_evaluated": len(valid_records),
        "total_matched": len(all_ranked_candidates),
        "returned_count": len(returned_candidates),
        "available_categories": available_categories,
        "failed_files": failed_files,
        "job_analysis": {
            "essential_skills": job_analysis.get("essential_skills", []),
            "optional_skills": job_analysis.get("optional_skills", []),
            "extracted_skills": job_analysis.get("extracted_skills", []),
            "by_category": job_analysis.get("by_category", {}),
            "warnings": job_analysis.get("warnings", []),
            "validation_messages": job_analysis.get("validation_messages", []),
        },
        "essential_skills": job_analysis.get("essential_skills", []),
        "optional_skills": job_analysis.get("optional_skills", []),
        "scoring_config_used": ranking_result.get("scoring_config_used", DEFAULT_SCORING_CONFIG),
        "validation_messages": all_validation_msgs,
        "ranked_candidates": returned_candidates,
    }


@api_router.post("/rank")
def rank_job_candidates(request: RankRequest):
    """
    Rank candidate resumes against the job description using dual-method
    TF-IDF text similarity and weighted skill match.
    Uses only candidates from the active screening session.
    Never falls back to any external or training datasets.
    """
    if not request.job_description or not request.job_description.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job description cannot be empty or whitespace-only.",
        )

    if not CURRENT_SCREENING_CANDIDATES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No candidate resumes have been uploaded for screening. Please upload candidate resumes first.",
        )

    # Build recruiter config if essential or optional skills were explicitly passed
    recruiter_config = None
    if request.essential_skills is not None or request.optional_skills is not None:
        recruiter_config = {
            "essential_skills": request.essential_skills or [],
            "optional_skills": request.optional_skills or [],
        }

    job_analysis = analyze_job_description(
        jd_text=request.job_description,
        recruiter_config=recruiter_config,
    )

    if not job_analysis.get("is_valid", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=job_analysis.get("validation_messages", ["Invalid job description."])[0],
        )

    # Build DataFrame purely from session candidates
    df_rows = []
    for cand in CURRENT_SCREENING_CANDIDATES.values():
        df_rows.append({
            "ID": cand["candidate_id"],
            "candidate_id": cand["candidate_id"],
            "filename": cand["filename"],
            "Category": cand.get("category", "Other / Unclassified"),
            "category": cand.get("category", "Other / Unclassified"),
            "raw_text": cand.get("raw_text", ""),
            "cleaned_text": cand.get("cleaned_text", cand.get("text_preview", "")),
            "skills": cand.get("skills", cand.get("all_skills", [])),
            "extracted_skills": cand.get("skills", cand.get("all_skills", [])),
            "is_usable": True,
            "is_duplicate": False,
            "word_count": cand.get("word_count", 0),
            "text_preview": cand.get("text_preview", ""),
            "file_size_formatted": cand.get("file_size_formatted", ""),
            "file_type": cand.get("file_type", "FILE"),
        })

    uploaded_df = pd.DataFrame(df_rows)

    ranking_result = rank_candidates(
        job_analysis=job_analysis,
        resumes_df=uploaded_df,
        top_n=None,
        scoring_config=request.scoring_config,
    )

    all_candidates = ranking_result.get("ranked_candidates", [])

    # Enrich metadata
    for c in all_candidates:
        fname = c.get("filename") or str(c.get("candidate_id", ""))
        matched = CURRENT_SCREENING_CANDIDATES.get(fname) or CURRENT_SCREENING_CANDIDATES.get(str(c["candidate_id"]))
        if matched:
            c["raw_text"] = matched.get("raw_text", "")
            c["cleaned_text"] = matched.get("cleaned_text", matched.get("text_preview", ""))
            c["all_skills"] = matched.get("all_skills", matched.get("skills", []))
            c["skills"] = c["all_skills"]
            c["extracted_skills"] = c["all_skills"]
            c["skill_count"] = len(c["all_skills"])
            c["word_count"] = matched.get("word_count", 0)
            c["text_preview"] = matched.get("text_preview", "")
            c["file_size_formatted"] = matched.get("file_size_formatted", "")
            c["file_type"] = matched.get("file_type", "FILE")
            c["category"] = matched.get("category", c.get("category", "Other / Unclassified"))
            # Update session candidate record with newly calculated ranking results
            matched.update(c)

    available_categories = sorted(list({c.get("category", "Other / Unclassified") for c in all_candidates}))

    # Category filter
    filtered = all_candidates
    if request.category_filter and request.category_filter.strip().upper() not in ("ALL", "ALL CATEGORIES", ""):
        filtered = [c for c in filtered if c.get("category", "").strip().lower() == request.category_filter.strip().lower()]

    # Min score filter
    if request.min_score and request.min_score > 0.0:
        filtered = [c for c in filtered if c["combined_score"] >= request.min_score]

    top_n = request.top_n or len(filtered)
    paged_candidates = filtered[:top_n]

    for idx, c in enumerate(paged_candidates):
        c["candidate_rank"] = idx + 1

    return {
        "is_valid": True,
        "total_evaluated": len(uploaded_df),
        "total_matched": len(all_candidates),
        "returned_count": len(paged_candidates),
        "available_categories": available_categories,
        "essential_skills": job_analysis.get("essential_skills", []),
        "optional_skills": job_analysis.get("optional_skills", []),
        "scoring_config_used": ranking_result.get("scoring_config_used", DEFAULT_SCORING_CONFIG),
        "validation_messages": ranking_result.get("validation_messages", []),
        "ranked_candidates": paged_candidates,
    }


@api_router.get("/candidate/{candidate_id}")
def get_candidate_details(candidate_id: str):
    """
    Retrieve candidate profile details for manual recruiter review.
    Fetches only from the current active screening session.
    Never searches external/Kaggle datasets.
    """
    candidate = CURRENT_SCREENING_CANDIDATES.get(str(candidate_id))
    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Candidate with ID '{candidate_id}' not found in current screening session.",
        )

    skills_set = parse_candidate_skills(candidate.get("all_skills") or candidate.get("skills", []))
    raw_text = str(candidate.get("raw_text", candidate.get("cleaned_text", candidate.get("text_preview", ""))))
    text_preview = candidate.get("text_preview") or (raw_text[:800].strip() + ("..." if len(raw_text) > 800 else "")) or "Not available."

    return {
        "candidate_id": str(candidate["candidate_id"]),
        "filename": str(candidate.get("filename", candidate["candidate_id"])),
        "category": str(candidate.get("category", "Other / Unclassified")),
        "skills": sorted(list(skills_set)),
        "all_skills": sorted(list(skills_set)),
        "skill_count": candidate.get("skill_count", len(skills_set)),
        "is_usable": bool(candidate.get("is_usable", True)),
        "text_preview": text_preview,
        "word_count": candidate.get("word_count", len(raw_text.split())),
        "file_size_formatted": candidate.get("file_size_formatted", ""),
        "file_type": candidate.get("file_type", "FILE"),
        "combined_score": candidate.get("combined_score"),
        "skill_match_score": candidate.get("skill_match_score"),
        "text_similarity_score": candidate.get("text_similarity_score"),
        "matched_essential": candidate.get("matched_essential", candidate.get("essential_skills_matched", [])),
        "missing_essential": candidate.get("missing_essential", candidate.get("essential_skills_missing", [])),
        "matched_optional": candidate.get("matched_optional", candidate.get("optional_skills_matched", [])),
        "missing_optional": candidate.get("missing_optional", candidate.get("optional_skills_missing", [])),
        "essential_coverage_percent": candidate.get("essential_coverage_percent"),
        "optional_coverage_percent": candidate.get("optional_coverage_percent"),
        "match_explanation": candidate.get("match_explanation", ""),
        "detailed_explanation": candidate.get("detailed_explanation", candidate.get("match_explanation", "")),
    }


# Include the API router both at /api and at root for full routing resilience on Vercel
app.include_router(api_router, prefix="/api")
app.include_router(api_router)


# ============================================================================
# STATIC FILES SERVING (FRONTEND)
# ============================================================================

frontend_path = PROJECT_ROOT / "frontend"
if not frontend_path.exists() or not (frontend_path / "index.html").exists():
    frontend_path = PROJECT_ROOT
if frontend_path.exists() and (frontend_path / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
