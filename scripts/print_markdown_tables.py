import json
from pathlib import Path

PROJECT_ROOT = Path(r"c:\Users\SKC\OneDrive\Documents\Desktop\TASK-3")
json_path = PROJECT_ROOT / "scripts" / "diagnostic_results.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for jd_name, jd_data in data.items():
    print(f"\n### Job Description: {jd_name}")
    print(f"- **Essential Skills**: {', '.join(jd_data['analysis']['essential_skills'])}")
    print(f"- **Optional Skills**: {', '.join(jd_data['analysis']['optional_skills'])}")
    print()
    print("| Rank | Resume Filename | Matched Essential Skills | Missing Essential Skills | Matched Optional Skills | Skill Score | Text Similarity | Combined Score |")
    print("| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: |")
    for c in jd_data["candidates"]:
        m_ess = c["Matched Essential Skills"] or "None"
        miss_ess = c["Missing Essential Skills"] or "None"
        m_opt = c["Matched Optional Skills"] or "None"
        print(f"| {c['Final Rank']} | `{c['Resume Filename']}` | {m_ess} | {miss_ess} | {m_opt} | {c['Skill Match Score']} | {c['Text Similarity Score']} | **{c['Combined Score']}** |")
