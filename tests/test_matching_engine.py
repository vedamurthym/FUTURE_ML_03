"""
Unit tests for Candidate Ranking and Matching Engine.

Tests:
- Resume matching all essential skills
- Resume missing an essential skill
- Optional skills affecting the score
- TF-IDF similarity calculation
- Correct combined score calculation
- Candidate ranking order
- Unusable resumes being excluded
- Duplicate copies being excluded
- Missing or empty skill fields
- Empty resume text
- Job description with no usable skills
- No essential skills and the documented fallback
- Recruiter-confirmed requirements taking precedence
- Deterministic ranking when scores tie
- Explanations matching actual calculated results
"""

import unittest
import pandas as pd
from src.matching_engine import (
    rank_candidates,
    compute_tfidf_similarities,
    compute_skill_metrics,
    parse_candidate_skills,
    DEFAULT_SCORING_CONFIG,
)
from src.job_analyzer import analyze_job_description


class TestMatchingEngine(unittest.TestCase):

    def setUp(self):
        # Sample structured job analysis with essential and optional skills
        self.sample_jd = """
        Job Title: Senior Data Engineer
        Requirements:
        - Must have deep experience with Python and SQL.
        - Docker is essential.
        Preferred Qualifications:
        - Experience with AWS is a plus.
        """
        self.job_analysis = analyze_job_description(self.sample_jd)

        # Synthetic small candidate DataFrame
        self.synthetic_resumes = pd.DataFrame([
            {
                "ID": 101,
                "Category": "INFORMATION-TECHNOLOGY",
                "is_usable": True,
                "is_duplicate": False,
                "cleaned_text": "expert python and sql developer with docker and aws cloud experience",
                "skills": "Python; SQL; Docker; AWS",
            },
            {
                "ID": 102,
                "Category": "INFORMATION-TECHNOLOGY",
                "is_usable": True,
                "is_duplicate": False,
                "cleaned_text": "data engineer skilled in python and sql databases",
                "skills": "Python; SQL",
            },
            {
                "ID": 103,
                "Category": "FINANCE",
                "is_usable": True,
                "is_duplicate": False,
                "cleaned_text": "accountant with excel and bookkeeping experience",
                "skills": "Excel",
            },
            {
                "ID": 104,
                "Category": "HR",
                "is_usable": False,  # Should be excluded
                "is_duplicate": False,
                "cleaned_text": "",
                "skills": "",
            },
            {
                "ID": 105,
                "Category": "INFORMATION-TECHNOLOGY",
                "is_usable": True,
                "is_duplicate": True,  # Duplicate of 101 - should be excluded
                "cleaned_text": "expert python and sql developer with docker and aws cloud experience",
                "skills": "Python; SQL; Docker; AWS",
            },
        ])

    def test_resume_matching_all_essential_skills(self):
        """Test candidate with 100% essential skills matched."""
        res = rank_candidates(self.job_analysis, self.synthetic_resumes)
        self.assertTrue(res["is_valid"])

        # Candidate 101 has Python, SQL, Docker (all 3 essential skills) + AWS
        top_cand = res["ranked_candidates"][0]
        self.assertEqual(top_cand["candidate_id"], 101)
        self.assertEqual(top_cand["essential_coverage_percent"], 100.0)
        self.assertEqual(len(top_cand["essential_skills_missing"]), 0)
        self.assertIn("Python", top_cand["essential_skills_matched"])
        self.assertIn("SQL", top_cand["essential_skills_matched"])
        self.assertIn("Docker", top_cand["essential_skills_matched"])

    def test_resume_missing_essential_skill(self):
        """Test candidate missing one of the essential skills."""
        res = rank_candidates(self.job_analysis, self.synthetic_resumes)
        # Candidate 102 has Python and SQL, but is missing Docker (2/3 = 66.67%)
        cand_102 = next(c for c in res["ranked_candidates"] if c["candidate_id"] == 102)

        self.assertAlmostEqual(cand_102["essential_coverage_percent"], 66.67, places=1)
        self.assertIn("Docker", cand_102["essential_skills_missing"])
        self.assertIn("Python", cand_102["essential_skills_matched"])
        self.assertIn("SQL", cand_102["essential_skills_matched"])

    def test_optional_skills_affecting_score(self):
        """Test that having optional skills yields a higher score when essential skills are equal."""
        # Create two candidates with identical essential skills (Python, SQL, Docker)
        # Candidate A has AWS (optional), Candidate B does not have AWS
        test_df = pd.DataFrame([
            {
                "ID": 201,
                "Category": "IT",
                "is_usable": True,
                "is_duplicate": False,
                "cleaned_text": "developer with python sql docker and aws",
                "skills": "Python; SQL; Docker; AWS",
            },
            {
                "ID": 202,
                "Category": "IT",
                "is_usable": True,
                "is_duplicate": False,
                "cleaned_text": "developer with python sql docker",
                "skills": "Python; SQL; Docker",
            },
        ])

        res = rank_candidates(self.job_analysis, test_df)
        c201 = res["ranked_candidates"][0]
        c202 = res["ranked_candidates"][1]

        self.assertEqual(c201["candidate_id"], 201)
        self.assertEqual(c202["candidate_id"], 202)
        # Candidate 201 has 100% optional (AWS), while Candidate 202 has 0% optional
        self.assertEqual(c201["optional_coverage_percent"], 100.0)
        self.assertEqual(c202["optional_coverage_percent"], 0.0)
        self.assertGreater(c201["skill_match_score"], c202["skill_match_score"])
        self.assertGreater(c201["combined_score"], c202["combined_score"])

    def test_tfidf_similarity_calculation(self):
        """Test TF-IDF cosine similarity computation."""
        jd_text = "python sql docker engineer"
        cand_texts = [
            "python sql docker engineer",         # Identical text -> ~100%
            "python developer with sql",           # Partial match
            "completely unrelated chef cooking",   # Zero overlap -> 0%
            "",                                    # Empty string -> 0%
        ]
        sims = compute_tfidf_similarities(jd_text, cand_texts)

        self.assertEqual(len(sims), 4)
        self.assertAlmostEqual(sims[0], 100.0, places=1)
        self.assertGreater(sims[1], 0.0)
        self.assertEqual(sims[2], 0.0)
        self.assertEqual(sims[3], 0.0)

    def test_correct_combined_score_calculation(self):
        """Verify the exact formula: combined = (0.7 * skill_score) + (0.3 * text_sim)."""
        custom_config = {
            "weight_skill": 0.60,
            "weight_text": 0.40,
            "essential_skill_weight": 0.75,
            "optional_skill_weight": 0.25,
        }
        res = rank_candidates(self.job_analysis, self.synthetic_resumes, scoring_config=custom_config)
        cand = res["ranked_candidates"][0]

        expected_combined = round(0.60 * cand["skill_match_score"] + 0.40 * cand["text_similarity_score"], 2)
        self.assertEqual(cand["combined_score"], expected_combined)

    def test_unusable_and_duplicate_resumes_excluded(self):
        """Verify that unusable (ID 104) and duplicate (ID 105) resumes are filtered out."""
        res = rank_candidates(self.job_analysis, self.synthetic_resumes)
        ranked_ids = [c["candidate_id"] for c in res["ranked_candidates"]]

        self.assertNotIn(104, ranked_ids, "Unusable resume (ID 104) was not excluded!")
        self.assertNotIn(105, ranked_ids, "Duplicate resume (ID 105) was not excluded!")
        self.assertEqual(len(ranked_ids), 3)

    def test_missing_or_empty_skill_fields(self):
        """Test handling of NaN, None, empty string, and malformed skill fields."""
        df_edge = pd.DataFrame([
            {"ID": 301, "Category": "IT", "is_usable": True, "is_duplicate": False, "cleaned_text": "python dev", "skills": None},
            {"ID": 302, "Category": "IT", "is_usable": True, "is_duplicate": False, "cleaned_text": "sql dev", "skills": float("nan")},
            {"ID": 303, "Category": "IT", "is_usable": True, "is_duplicate": False, "cleaned_text": "docker dev", "skills": ""},
            {"ID": 304, "Category": "IT", "is_usable": True, "is_duplicate": False, "cleaned_text": "dev", "skills": ["Python", "SQL"]},
        ])
        res = rank_candidates(self.job_analysis, df_edge)
        self.assertTrue(res["is_valid"])
        self.assertEqual(len(res["ranked_candidates"]), 4)

    def test_empty_resume_text(self):
        """Test candidate with empty cleaned_text."""
        df_empty_text = pd.DataFrame([
            {"ID": 401, "Category": "IT", "is_usable": True, "is_duplicate": False, "cleaned_text": "", "skills": "Python"},
        ])
        res = rank_candidates(self.job_analysis, df_empty_text)
        cand = res["ranked_candidates"][0]
        self.assertEqual(cand["text_similarity_score"], 0.0)
        self.assertGreater(cand["skill_match_score"], 0.0)

    def test_job_description_with_no_usable_skills(self):
        """Test handling when a vague job description has no detected skills."""
        vague_jd = "Small office looking for someone to help during business hours."
        vague_analysis = analyze_job_description(vague_jd)

        res = rank_candidates(vague_analysis, self.synthetic_resumes)
        self.assertTrue(res["is_valid"])
        # Should contain notice in validation messages
        has_notice = any("no essential or optional skill requirements" in m for m in res["validation_messages"])
        self.assertTrue(has_notice)

    def test_no_essential_skills_documented_fallback(self):
        """Test fallback when JD has only optional skills."""
        opt_only_analysis = {
            "is_valid": True,
            "has_skills": True,
            "cleaned_jd": "python and aws developer",
            "essential_skills": [],
            "optional_skills": ["Python", "AWS"],
            "validation_messages": [],
        }
        res = rank_candidates(opt_only_analysis, self.synthetic_resumes)
        cand = res["ranked_candidates"][0]

        self.assertIsNone(cand["essential_coverage_percent"])
        self.assertIsNotNone(cand["optional_coverage_percent"])
        # Skill match score uses optional coverage as fallback
        self.assertEqual(cand["skill_match_score"], cand["optional_coverage_percent"])

    def test_recruiter_confirmed_requirements_precedence(self):
        """Test that recruiter configuration takes precedence through analyze_job_description."""
        recruiter_config = {
            "essential_skills": ["Excel"],
            "optional_skills": ["SQL"],
        }
        config_analysis = analyze_job_description(self.sample_jd, recruiter_config=recruiter_config)
        res = rank_candidates(config_analysis, self.synthetic_resumes)

        # Under this recruiter config, Candidate 103 (Finance / Excel) has 100% essential!
        top_cand = res["ranked_candidates"][0]
        self.assertEqual(top_cand["candidate_id"], 103)
        self.assertEqual(top_cand["essential_coverage_percent"], 100.0)
        self.assertIn("Excel", top_cand["essential_skills_matched"])

    def test_deterministic_ranking_on_score_tie(self):
        """Test deterministic tie-breaking when two candidates have identical scores."""
        tied_df = pd.DataFrame([
            {"ID": 902, "Category": "IT", "is_usable": True, "is_duplicate": False, "cleaned_text": "python dev", "skills": "Python"},
            {"ID": 901, "Category": "IT", "is_usable": True, "is_duplicate": False, "cleaned_text": "python dev", "skills": "Python"},
        ])
        res1 = rank_candidates(self.job_analysis, tied_df)
        res2 = rank_candidates(self.job_analysis, tied_df)

        ids1 = [c["candidate_id"] for c in res1["ranked_candidates"]]
        ids2 = [c["candidate_id"] for c in res2["ranked_candidates"]]

        # Both runs must produce identical ordering, and tie-break by ID ascending (901 before 902)
        self.assertEqual(ids1, ids2)
        self.assertEqual(ids1, [901, 902])

    def test_explanations_match_calculated_results(self):
        """Verify that explanation strings accurately describe the calculated metrics."""
        res = rank_candidates(self.job_analysis, self.synthetic_resumes)
        cand_102 = next(c for c in res["ranked_candidates"] if c["candidate_id"] == 102)

        exp = cand_102["match_explanation"]
        self.assertIn("Matched 2 of 3 essential skills", exp)
        self.assertIn("Missing Docker", exp)
        self.assertIn("Text similarity was", exp)
        self.assertIn("Review the resume manually to verify proficiency.", exp)


if __name__ == "__main__":
    unittest.main()
