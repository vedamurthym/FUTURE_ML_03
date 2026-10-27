"""
Unit tests for text preprocessing module.
Tests normal text, empty text, whitespace-only text, punctuation removal,
technical terms preservation (C++, C#, .NET, Node.js), and duplicate handling.
"""

import unittest
import pandas as pd
from src.text_preprocessor import (
    clean_text,
    extract_text_from_html,
    resolve_resume_text,
    identify_duplicates,
    preprocess_dataframe,
    get_modeling_dataset,
)


class TestTextPreprocessor(unittest.TestCase):

    def test_normal_text(self):
        """Test basic text normalization and lowercasing."""
        raw = "Senior Software Engineer with 5+ years of experience in Python."
        cleaned = clean_text(raw)
        self.assertIn("senior software engineer", cleaned)
        self.assertIn("python", cleaned)
        self.assertEqual(cleaned, cleaned.lower())
        self.assertNotIn("\n", cleaned)
        self.assertNotIn("  ", cleaned)

    def test_empty_text(self):
        """Test handling of None and empty string."""
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(None), "")
        self.assertEqual(extract_text_from_html(""), "")
        self.assertEqual(extract_text_from_html(None), "")

    def test_whitespace_only_text(self):
        """Test handling of whitespace-only strings."""
        self.assertEqual(clean_text("   \n\t  \r  "), "")
        raw_text, cleaned_text, is_usable = resolve_resume_text("   \n\t  ", "")
        self.assertFalse(is_usable)
        self.assertEqual(cleaned_text, "")

    def test_punctuation_removal(self):
        """Test stripping of standard punctuation, bullets, and symbols."""
        raw = "• Bullet point one! 'Quoted text' -- semi-colon; & comma, tests?"
        cleaned = clean_text(raw)
        self.assertNotIn("•", cleaned)
        self.assertNotIn("!", cleaned)
        self.assertNotIn(";", cleaned)
        self.assertNotIn("?", cleaned)
        self.assertNotIn("--", cleaned)
        self.assertEqual(cleaned, "bullet point one quoted text semi colon comma tests")

    def test_technical_terms_preservation(self):
        """Test preservation of technical terms: C++, C#, .NET, Node.js, etc."""
        raw = "Proficient in C++, C#, .NET framework, and Node.js. Also uses ASP.NET and CI/CD."
        cleaned = clean_text(raw)
        
        # Verify specific technical tokens are intact
        self.assertIn("c++", cleaned)
        self.assertIn("c#", cleaned)
        self.assertIn(".net", cleaned)
        self.assertIn("node.js", cleaned)
        self.assertIn("asp.net", cleaned)
        self.assertIn("ci/cd", cleaned)
        
        # Verify leading .NET is preserved
        leading_dotnet = clean_text(".NET Developer skilled in C++ and C#.")
        self.assertTrue(leading_dotnet.startswith(".net"))
        self.assertIn("c++", leading_dotnet)
        self.assertIn("c#", leading_dotnet)

    def test_html_extraction_and_fallback(self):
        """Test HTML text extraction and fallback when resume_str is missing."""
        # 1. Valid HTML fallback
        html_content = "<div><h1>Developer Profile</h1><p>Skilled in Java and SQL.</p></div>"
        raw_text, cleaned_text, is_usable = resolve_resume_text("", html_content)
        self.assertTrue(is_usable)
        self.assertIn("developer profile", cleaned_text)
        self.assertIn("skilled in java and sql", cleaned_text)

        # 2. Empty HTML markup (like row 656)
        empty_html = '<div class="section" id="SEC1"> </div><div id="SEC2"> </div>'
        raw_empty, cleaned_empty, is_usable_empty = resolve_resume_text("", empty_html)
        self.assertFalse(is_usable_empty)
        self.assertEqual(cleaned_empty, "")

    def test_duplicate_handling(self):
        """Test that duplicate resume texts are detected and flagged without row deletion."""
        df = pd.DataFrame([
            {"ID": 101, "Category": "IT", "is_usable": True, "cleaned_text": "expert in python and aws"},
            {"ID": 102, "Category": "IT", "is_usable": True, "cleaned_text": "frontend react developer"},
            {"ID": 103, "Category": "IT", "is_usable": True, "cleaned_text": "expert in python and aws"}, # Duplicate of 101
            {"ID": 104, "Category": "HR", "is_usable": False, "cleaned_text": ""},
        ])
        
        df["is_duplicate"] = identify_duplicates(df, text_column="cleaned_text")
        
        # Row 0 is first occurrence -> False
        self.assertFalse(df.loc[0, "is_duplicate"])
        # Row 1 is unique -> False
        self.assertFalse(df.loc[1, "is_duplicate"])
        # Row 2 is second occurrence of row 0 text -> True
        self.assertTrue(df.loc[2, "is_duplicate"])
        # Row 3 is unusable -> False
        self.assertFalse(df.loc[3, "is_duplicate"])
        
        # Verify deduplication for modeling subset
        model_df = get_modeling_dataset(df)
        self.assertEqual(len(model_df), 2)
        self.assertListEqual(list(model_df["ID"]), [101, 102])

    def test_full_pipeline_preserves_id_and_category(self):
        """Test that preprocess_dataframe preserves ID and Category for all records."""
        mini_df = pd.DataFrame([
            {"ID": 1001, "Category": "FINANCE", "Resume_str": "Accountant with Excel skills", "Resume_html": ""},
            {"ID": 1002, "Category": "ENGINEERING", "Resume_str": "   ", "Resume_html": "<p>Embedded C++ Engineer</p>"},
            {"ID": 1003, "Category": "SALES", "Resume_str": "", "Resume_html": "<div> </div>"},
        ])
        
        processed = preprocess_dataframe(mini_df)
        self.assertEqual(len(processed), 3)
        self.assertListEqual(list(processed.columns), [
            "ID", "Category", "raw_text", "cleaned_text", "is_usable", "is_duplicate"
        ])
        self.assertTrue(processed.loc[0, "is_usable"])
        self.assertTrue(processed.loc[1, "is_usable"])
        self.assertIn("c++", processed.loc[1, "cleaned_text"])
        self.assertFalse(processed.loc[2, "is_usable"])


if __name__ == "__main__":
    unittest.main()
