"""
Unit tests for explainable NLP skill extractor.
Covers:
- Python and SQL extraction
- C and R false-positive prevention vs true-positive detection
- C++, C#, .NET, Node.js, PL/SQL, and CI/CD preservation
- Case-insensitive matching
- Aliases (e.g. sklearn -> scikit-learn, postgres -> PostgreSQL, k8s -> Kubernetes, aws -> AWS)
- Missing or empty resume text
- Duplicate and unusable record filtering
- Evidence snippets and category grouping
"""

import unittest
import pandas as pd
from src.skill_extractor import (
    extract_skills,
    process_resumes_skills,
    make_snippet,
    DISCLAIMER_TEXT,
)


class TestSkillExtractor(unittest.TestCase):

    def test_python_and_sql_extraction(self):
        """Test accurate extraction of Python and SQL."""
        text = "5 years of experience developing ETL pipelines in Python and writing complex SQL queries."
        result = extract_skills(text)
        
        self.assertIn("Python", result["skills"])
        self.assertIn("SQL", result["skills"])
        self.assertIn("Programming Languages", result["categories"])
        self.assertIn("Python", result["by_category"]["Programming Languages"])
        self.assertIn("SQL", result["by_category"]["Programming Languages"])

    def test_c_and_r_false_positive_prevention(self):
        """Test that single-letter languages (C, R) avoid false positives on ordinary text."""
        # False positive test cases for C
        non_c_texts = [
            "Patient was administered 500mg of Vitamin C daily.",
            "Achieved Grade C in linear algebra.",
            "Traveled to Washington, D.C. for a company conference.",
            "Option C was approved by the board.",
            "Worked at C-level executive management.",
        ]
        for text in non_c_texts:
            result = extract_skills(text)
            self.assertNotIn("C", result["skills"], f"False positive for C on text: '{text}'")

        # True positive test cases for C
        true_c_texts = [
            "Embedded firmware developer proficient in C and Python.",
            "Core skills: C, C++, and Java.",
            "Extensive experience with embedded C programming.",
            "High-performance systems using C/C++.",
        ]
        for text in true_c_texts:
            result = extract_skills(text)
            self.assertIn("C", result["skills"], f"Failed to match C on text: '{text}'")

        # False positive test cases for R
        non_r_texts = [
            "Senior specialist in the R&D division.",
            "Purchased supplies from Toys R Us.",
            "The R factor was calculated according to standard tables.",
            "Managed Model R product launch.",
        ]
        for text in non_r_texts:
            result = extract_skills(text)
            self.assertNotIn("R", result["skills"], f"False positive for R on text: '{text}'")

        # True positive test cases for R
        true_r_texts = [
            "Data analysis using R and Python.",
            "Built predictive models in R language and RStudio.",
            "Statistical computing with R, Python, and SQL.",
            "Created packages with R scripts for econometric analysis.",
        ]
        for text in true_r_texts:
            result = extract_skills(text)
            self.assertIn("R", result["skills"], f"Failed to match R on text: '{text}'")

        # False positive test for polysemous words (Swift, Rust)
        self.assertNotIn("Swift", extract_skills("Ensured the swift resolution of client complaints.")["skills"])
        self.assertIn("Swift", extract_skills("iOS engineer skilled in Swift and Objective-C.")["skills"])
        self.assertNotIn("Rust", extract_skills("Treated equipment to prevent rust and corrosion.")["skills"])
        self.assertIn("Rust", extract_skills("Systems programming with Rust and C++.")["skills"])

    def test_technical_symbols_preservation(self):
        """Test preservation of technical names containing symbols (C++, C#, .NET, Node.js, PL/SQL, CI/CD)."""
        text = (
            "Senior full-stack engineer experienced in C++, C#, .NET core, "
            "Node.js microservices, PL/SQL stored procedures, and CI/CD pipelines."
        )
        result = extract_skills(text)
        
        self.assertIn("C++", result["skills"])
        self.assertIn("C#", result["skills"])
        self.assertIn(".NET", result["skills"])
        self.assertIn("Node.js", result["skills"])
        self.assertIn("PL/SQL", result["skills"])
        self.assertIn("CI/CD", result["skills"])

    def test_case_insensitive_matching(self):
        """Test that matching handles uppercase, lowercase, and mixed case."""
        cases = [
            "python and sql developer",
            "PYTHON AND SQL DEVELOPER",
            "PyThOn AnD SqL dEvElOpEr",
        ]
        for text in cases:
            res = extract_skills(text)
            self.assertIn("Python", res["skills"])
            self.assertIn("SQL", res["skills"])

    def test_aliases(self):
        """Test that skill aliases resolve to standard canonical skill names."""
        # scikit-learn / sklearn
        res1 = extract_skills("Trained Random Forest with sklearn and pandas.")
        self.assertIn("scikit-learn", res1["skills"])
        
        # PostgreSQL / postgres
        res2 = extract_skills("Database administration with postgres and redis.")
        self.assertIn("PostgreSQL", res2["skills"])
        self.assertIn("Redis", res2["skills"])

        # Kubernetes / k8s
        res3 = extract_skills("Container orchestration using k8s and docker.")
        self.assertIn("Kubernetes", res3["skills"])
        self.assertIn("Docker", res3["skills"])

        # AWS / Amazon Web Services
        res4 = extract_skills("Cloud architecture deployed on Amazon Web Services.")
        self.assertIn("AWS", res4["skills"])

        # TypeScript / TS
        res5 = extract_skills("Frontend built with React and TypeScript.")
        self.assertIn("TypeScript", res5["skills"])
        self.assertIn("React", res5["skills"])

    def test_missing_and_empty_text(self):
        """Test that empty, whitespace, and None return safe empty results."""
        for val in ["", None, "    \n\t   "]:
            res = extract_skills(val)
            self.assertEqual(res["skills"], [])
            self.assertEqual(res["categories"], [])
            self.assertEqual(res["by_category"], {})
            self.assertEqual(res["evidence"], {})
            self.assertEqual(res["total_matches"], 0)

    def test_evidence_snippets_and_category_grouping(self):
        """Test snippet extraction and category grouping structures."""
        text = "Strong leadership and project management skills demonstrated when managing teams."
        res = extract_skills(text, window_size=30)
        
        self.assertIn("Leadership", res["skills"])
        self.assertIn("Project Management", res["skills"])
        self.assertIn("Professional Skills", res["categories"])
        
        # Verify evidence structure
        lead_evidence = res["evidence"]["Leadership"]
        self.assertGreater(len(lead_evidence), 0)
        first_ev = lead_evidence[0]
        self.assertIn("matched_phrase", first_ev)
        self.assertIn("snippet", first_ev)
        self.assertIn("start", first_ev)
        self.assertIn("end", first_ev)
        self.assertIn("leadership", first_ev["matched_phrase"].lower())
        self.assertIn("leadership", first_ev["snippet"].lower())

    def test_duplicate_and_unusable_record_filtering(self):
        """Test process_resumes_skills on a synthetic dataframe with usable, unusable, and duplicate flags."""
        df = pd.DataFrame([
            {
                "ID": 1,
                "Category": "IT",
                "is_usable": True,
                "is_duplicate": False,
                "raw_text": "Experienced Python and Docker developer.",
            },
            {
                "ID": 2,
                "Category": "HR",
                "is_usable": False,  # Unusable record
                "is_duplicate": False,
                "raw_text": "",
            },
            {
                "ID": 3,
                "Category": "IT",
                "is_usable": True,
                "is_duplicate": True,  # Duplicate of record 1
                "raw_text": "Experienced Python and Docker developer.",
            },
        ])

        processed = process_resumes_skills(df)
        self.assertEqual(len(processed), 3)
        self.assertEqual(processed.loc[0, "skill_count"], 2)
        self.assertIn("Python", processed.loc[0, "skills"])
        self.assertIn("Docker", processed.loc[0, "skills"])

        # Unusable record should yield 0 skills
        self.assertEqual(processed.loc[1, "skill_count"], 0)
        self.assertEqual(processed.loc[1, "skills"], "")

        # Verify disclaimer text is present
        self.assertEqual(processed.loc[0, "disclaimer"], DISCLAIMER_TEXT)

        # Modeling/Evaluation filter: usable and non-duplicate only
        eval_subset = processed[processed["is_usable"] & (~processed["is_duplicate"])]
        self.assertEqual(len(eval_subset), 1)
        self.assertEqual(eval_subset.iloc[0]["ID"], 1)


if __name__ == "__main__":
    unittest.main()
