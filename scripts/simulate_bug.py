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

# Step 1: Run Job 1 (ML Engineer)
job1_analysis = analyze_job_description(JD_TEXTS["Junior Machine Learning Engineer"])
job1_res = rank_candidates(job_analysis=job1_analysis, resumes_df=resumes_df)

print("=== JOB 1 RESULTS (Normal ML Engineer) ===")
for c in job1_res["ranked_candidates"][:5]:
    print(f"Rank {c['candidate_rank']}: {c['filename']} - Combined: {c['combined_score']} (Skill: {c['skill_match_score']}, Text: {c['text_similarity_score']})")

# Step 2: Simulate STALE skills bug when running Job 2 (Data Analyst) with Job 1's essential/optional skills
stale_config_job1 = {
    "essential_skills": job1_analysis["essential_skills"],
    "optional_skills": job1_analysis["optional_skills"]
}
job2_stale_analysis = analyze_job_description(JD_TEXTS["Data Analyst / Junior Data Scientist"], recruiter_config=stale_config_job1)
job2_stale_res = rank_candidates(job_analysis=job2_stale_analysis, resumes_df=resumes_df)

print("\n=== JOB 2 RESULTS WITH STALE JOB 1 SKILLS (Simulating UI bug) ===")
for c in job2_stale_res["ranked_candidates"][:5]:
    print(f"Rank {c['candidate_rank']}: {c['filename']} - Combined: {c['combined_score']} (Skill: {c['skill_match_score']}, Text: {c['text_similarity_score']})")

# Step 3: Simulate STALE skills bug when running Job 3 (Backend Developer) with Job 1's essential/optional skills
job3_stale_analysis = analyze_job_description(JD_TEXTS["Python Backend Developer Intern"], recruiter_config=stale_config_job1)
job3_stale_res = rank_candidates(job_analysis=job3_stale_analysis, resumes_df=resumes_df)

print("\n=== JOB 3 RESULTS WITH STALE JOB 1 SKILLS (Simulating UI bug) ===")
for c in job3_stale_res["ranked_candidates"][:5]:
    print(f"Rank {c['candidate_rank']}: {c['filename']} - Combined: {c['combined_score']} (Skill: {c['skill_match_score']}, Text: {c['text_similarity_score']})")

# Step 4: What if the entire previous request was cached or state.jobDescription was reused?
# In frontend rankCandidates():
# job_description: state.jobDescription || dom.jdInput.value.trim()
# If state.jobDescription is NOT reset, /api/rank literally sends Job 1's description!
print("\n=== REUSED state.jobDescription (Job 1 rerun) ===")
for c in job1_res["ranked_candidates"][:5]:
    print(f"Rank {c['candidate_rank']}: {c['filename']} - Combined: {c['combined_score']}")
