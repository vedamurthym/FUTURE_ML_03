import json
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_parser import parse_single_resume
from src.job_analyzer import analyze_job_description
from src.matching_engine import rank_candidates
from scripts.run_full_diagnosis import JD_TEXTS

resumes_dir = PROJECT_ROOT / "data" / "test_resumes"
parsed_resumes = []
for i in range(1, 21):
    fname = f"resume_{i:02d}.pdf"
    fpath = resumes_dir / fname
    with open(fpath, "rb") as f:
        content = f.read()
    parsed_resumes.append(parse_single_resume(fname, content))

df_rows = []
for item in parsed_resumes:
    df_rows.append({
        "ID": item["filename"],
        "candidate_id": item["filename"],
        "filename": item["filename"],
        "raw_text": item["raw_text"],
        "cleaned_text": item["cleaned_text"],
        "skills": item["extracted_skills"],
        "extracted_skills": item["extracted_skills"],
        "is_usable": True,
        "is_duplicate": False,
        "word_count": item["word_count"],
        "text_preview": item["text_preview"],
        "file_size_formatted": item.get("file_size_formatted", ""),
        "file_type": item.get("file_type", "PDF"),
    })
resumes_df = pd.DataFrame(df_rows)

tables = {}
for jd_name, jd_text in JD_TEXTS.items():
    analysis = analyze_job_description(jd_text)
    res = rank_candidates(job_analysis=analysis, resumes_df=resumes_df)
    
    rows = []
    for c in res["ranked_candidates"]:
        rows.append({
            "Resume Filename": c["filename"],
            "Matched Essential Skills": ", ".join(c["matched_essential"]) if c["matched_essential"] else "None",
            "Missing Essential Skills": ", ".join(c["missing_essential"]) if c["missing_essential"] else "None",
            "Matched Optional Skills": ", ".join(c["matched_optional"]) if c["matched_optional"] else "None",
            "Skill Match Score": f"{c['skill_match_score']:.2f}%",
            "Text Similarity Score": f"{c['text_similarity_score']:.2f}%",
            "Combined Score": f"{c['combined_score']:.2f}%",
            "Final Rank": c["candidate_rank"],
        })
    tables[jd_name] = {
        "analysis": analysis,
        "candidates": rows,
    }

# Save as json for easy artifact / report generation
with open(PROJECT_ROOT / "scripts" / "diagnostic_results.json", "w", encoding="utf-8") as f:
    json.dump(tables, f, indent=2)

print("Saved diagnostic_results.json successfully!")
