"""
Demonstration Script for Stage 4: Job Description Analysis.

Analyzes fictional job descriptions to demonstrate:
1. Automatic extraction and parsing of required vs. preferred vs. general skills.
2. Recruiter-provided configuration with confirmed requirements.
3. Reporting and validation of unknown / unrecognized skills.
4. Handling edge cases (descriptions with no detectable skills, empty descriptions).

Uses strictly fictional data; no candidate personal data is saved or displayed.
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.job_analyzer import analyze_job_description


FICTIONAL_JD = """
Job Title: Senior Machine Learning & Backend Engineer
Company: NovaTech AI Labs (Fictional)
Location: Remote / Bengaluru, India

About Us:
NovaTech AI Labs builds next-generation intelligent automation platforms for enterprise workflows.

Core Requirements:
- 4+ years of hands-on software development in Python and SQL.
- Must have solid experience with scikit-learn, PyTorch, and Machine Learning pipelines.
- Deep familiarity with Docker and Linux server administration.
- Strong problem-solving ability and engineering discipline.

Preferred Qualifications:
- Experience with AWS cloud infrastructure and CI/CD pipelines is preferred.
- Familiarity with FastAPI or Flask is a plus.
- Knowledge of Kubernetes and Apache Spark is nice to have.

Professional Attributes:
- Excellent communication and leadership skills.
- Collaborative mindset and passion for mentoring junior engineers.
"""

FICTIONAL_VAGUE_JD = """
Job Title: General Office Helper
We need someone friendly and energetic to join our small team.
The ideal candidate arrives at the office on time and helps with daily tasks.
Please send your cover letter to apply.
"""


def print_section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def format_analysis_output(res: dict):
    print(f" -> Is Valid          : {res['is_valid']}")
    print(f" -> Skills Detected   : {res['has_skills']} ({len(res['extracted_skills'])} unique skills)")
    print(f" -> All Extracted     : {', '.join(res['extracted_skills']) if res['extracted_skills'] else 'None'}")
    
    print("\n -> Skills Grouped by Category:")
    if res["by_category"]:
        for cat, skills in res["by_category"].items():
            print(f"     * {cat:<24}: {', '.join(skills)}")
    else:
        print("     * None")

    print("\n -> Requirement Breakdown:")
    print(f"     * Essential Skills  : {', '.join(res['essential_skills']) if res['essential_skills'] else 'None'}")
    print(f"     * Optional Skills   : {', '.join(res['optional_skills']) if res['optional_skills'] else 'None'}")
    print(f"     * Unspecified Skills: {', '.join(res['unspecified_skills']) if res['unspecified_skills'] else 'None'}")

    if res["unknown_skills"]:
        print("\n -> Unrecognized Recruiter Skills:")
        for unk in res["unknown_skills"]:
            print(f"     ! [{unk['specified_as'].upper()}] '{unk['raw_skill']}': {unk['message']}")

    if res["validation_messages"]:
        print("\n -> Validation Notes & Guidance:")
        for msg in res["validation_messages"]:
            print(f"     - {msg}")


def main():
    print_section("STAGE 4: JOB DESCRIPTION ANALYSIS DEMONSTRATION")

    # Demo 1: Automatic Heuristic Analysis
    print_section("DEMO 1: Automatic Heuristic Analysis (No Recruiter Config)")
    print("Fictional JD Snippet: Senior Machine Learning & Backend Engineer")
    res1 = analyze_job_description(FICTIONAL_JD)
    format_analysis_output(res1)

    # Demo 2: Recruiter-Provided Configuration with Precedence and Unknown Skills
    print_section("DEMO 2: Recruiter Configuration (With Known & Unknown Skills)")
    recruiter_config = {
        "essential_skills": ["Python", "scikit-learn", "Docker", "QuantumHyperDB_v2"],
        "optional_skills": ["AWS", "FastAPI", "MegaTool9000"],
    }
    print(f"Recruiter Supplied Config:")
    print(f" - Essential inputs: {recruiter_config['essential_skills']}")
    print(f" - Optional inputs : {recruiter_config['optional_skills']}")
    
    res2 = analyze_job_description(FICTIONAL_JD, recruiter_config=recruiter_config)
    format_analysis_output(res2)

    # Demo 3: Job Description with No Detectable Skills
    print_section("DEMO 3: Edge Case - Job Description with No Detectable Skills")
    res3 = analyze_job_description(FICTIONAL_VAGUE_JD)
    format_analysis_output(res3)

    # Demo 4: Empty / Whitespace-Only Job Description
    print_section("DEMO 4: Edge Case - Empty / Whitespace-Only Description")
    res4 = analyze_job_description("    \n\t   ")
    format_analysis_output(res4)

    print("\n" + "=" * 70)
    print("STAGE 4 DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
