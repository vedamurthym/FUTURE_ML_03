"""
Unit tests for Job Description Analysis module.
Covers:
- Normal job description analysis
- Empty and whitespace-only descriptions
- Required versus preferred skill handling
- Skills grouped by category
- Unknown recruiter-supplied skills reporting
- Alias handling in recruiter configuration
- Direct reuse of existing skill extractor taxonomy
- Handling of descriptions with no detectable skills
- Recruiter essential and optional skill configuration precedence
"""

import unittest
from src.job_analyzer import (
    validate_job_description,
    resolve_recruiter_skill,
    parse_skills_by_section,
    validate_recruiter_configuration,
    analyze_job_description,
)


class TestJobAnalyzer(unittest.TestCase):

    def setUp(self):
        self.sample_jd = """
        Job Title: Senior Backend & Data Engineer
        Company: CloudTech Innovations

        About the Role:
        We are building scalable distributed data systems.

        Requirements:
        - 4+ years of professional programming in Python and SQL.
        - Must have deep experience with Docker and Linux environments.

        Preferred Qualifications:
        - Familiarity with AWS and Apache Spark is a plus.
        - Experience with Kubernetes and CI/CD pipelines is preferred.

        Other Details:
        - Excellent communication and problem-solving skills.
        """

    def test_normal_job_description(self):
        """Test standard job description analysis and field extraction."""
        result = analyze_job_description(self.sample_jd)

        self.assertTrue(result["is_valid"])
        self.assertTrue(result["has_skills"])
        self.assertIn("Python", result["extracted_skills"])
        self.assertIn("SQL", result["extracted_skills"])
        self.assertIn("Docker", result["extracted_skills"])
        self.assertIn("AWS", result["extracted_skills"])
        self.assertIn("Communication", result["extracted_skills"])
        self.assertGreater(len(result["cleaned_jd"]), 0)

    def test_empty_and_whitespace_descriptions(self):
        """Test validation and handling of None, empty, and whitespace-only JDs."""
        # None input
        res_none = analyze_job_description(None)
        self.assertFalse(res_none["is_valid"])
        self.assertFalse(res_none["has_skills"])
        self.assertIn("None", res_none["validation_messages"][0])

        # Empty string
        res_empty = analyze_job_description("")
        self.assertFalse(res_empty["is_valid"])
        self.assertFalse(res_empty["has_skills"])

        # Whitespace-only string
        res_ws = analyze_job_description("    \n\t   \r  ")
        self.assertFalse(res_ws["is_valid"])
        self.assertFalse(res_ws["has_skills"])
        self.assertIn("whitespace-only", res_ws["validation_messages"][0])

    def test_required_vs_preferred_skill_handling(self):
        """Test that skills are appropriately separated into essential and optional automatically."""
        result = analyze_job_description(self.sample_jd)

        auto_req = result["auto_suggested_essential"]
        auto_pref = result["auto_suggested_optional"]
        unspecified = result["unspecified_skills"]

        # Skills in 'Requirements:' section
        self.assertIn("Python", auto_req)
        self.assertIn("SQL", auto_req)
        self.assertIn("Docker", auto_req)
        self.assertIn("Linux", auto_req)

        # Skills in 'Preferred Qualifications:' section
        self.assertIn("AWS", auto_pref)
        self.assertIn("Apache Spark", auto_pref)
        self.assertIn("Kubernetes", auto_pref)
        self.assertIn("CI/CD", auto_pref)

        # Skills in 'Other Details:' section should not be assumed essential
        self.assertIn("Communication", unspecified)
        self.assertIn("Problem Solving", unspecified)
        self.assertNotIn("Communication", auto_req)

    def test_skills_grouped_by_category(self):
        """Test category groupings within the structured output."""
        result = analyze_job_description(self.sample_jd)
        by_cat = result["by_category"]

        self.assertIn("Programming Languages", by_cat)
        self.assertIn("Python", by_cat["Programming Languages"])
        self.assertIn("SQL", by_cat["Programming Languages"])

        self.assertIn("Databases & Dev Tools", by_cat)
        self.assertIn("Docker", by_cat["Databases & Dev Tools"])
        self.assertIn("Linux", by_cat["Databases & Dev Tools"])
        self.assertIn("AWS", by_cat["Databases & Dev Tools"])

        self.assertIn("Professional Skills", by_cat)
        self.assertIn("Communication", by_cat["Professional Skills"])

    def test_unknown_recruiter_supplied_skills(self):
        """Test that unknown recruiter skills are reported instead of silently discarded."""
        config = {
            "essential_skills": ["Python", "QuantumHyperDB", "SQL"],
            "optional_skills": ["SuperSecretTool123", "Docker"],
        }
        result = analyze_job_description(self.sample_jd, recruiter_config=config)

        self.assertIn("Python", result["confirmed_essential"])
        self.assertIn("SQL", result["confirmed_essential"])
        self.assertIn("Docker", result["confirmed_optional"])

        unknown_raw = [u["raw_skill"] for u in result["unknown_skills"]]
        self.assertIn("QuantumHyperDB", unknown_raw)
        self.assertIn("SuperSecretTool123", unknown_raw)

        # Check validation warning message exists
        has_warning = any("not recognized" in msg for msg in result["validation_messages"])
        self.assertTrue(has_warning)

    def test_alias_handling_in_configuration(self):
        """Test that recruiter-provided aliases resolve to canonical taxonomy names."""
        config = {
            "essential_skills": ["k8s", "sklearn", "postgres"],
            "optional_skills": ["Nodejs", "AWS"],
        }
        confirmed_req, confirmed_opt, unknown = validate_recruiter_configuration(config)

        self.assertIn("Kubernetes", confirmed_req)
        self.assertIn("scikit-learn", confirmed_req)
        self.assertIn("PostgreSQL", confirmed_req)
        self.assertIn("Node.js", confirmed_opt)
        self.assertIn("AWS", confirmed_opt)
        self.assertEqual(len(unknown), 0)

    def test_reuse_of_existing_skill_extractor(self):
        """Verify that resolve_recruiter_skill uses the exact same taxonomy as src.skill_extractor."""
        from src.skill_extractor import SKILL_DEFINITIONS

        # Pick random skills from SKILL_DEFINITIONS and test resolution
        sample_canonical = [s["name"] for s in SKILL_DEFINITIONS[:5]]
        for name in sample_canonical:
            resolved = resolve_recruiter_skill(name)
            self.assertEqual(resolved, name)

    def test_no_detectable_skills_message(self):
        """Test handling of job descriptions with valid prose but no detectable skills."""
        vague_jd = (
            "We are seeking an energetic individual to join our team. "
            "The candidate should come to the office every morning from 9am to 5pm. "
            "Please apply with a cover letter."
        )
        result = analyze_job_description(vague_jd)

        self.assertTrue(result["is_valid"])
        self.assertFalse(result["has_skills"])
        self.assertEqual(result["extracted_skills"], [])
        self.assertEqual(result["essential_skills"], [])
        self.assertEqual(result["optional_skills"], [])

        # Check for helpful recruiter guidance message
        has_guidance = any("No technical or professional skills were detected" in msg for msg in result["validation_messages"])
        self.assertTrue(has_guidance)

    def test_recruiter_configuration_precedence(self):
        """Test that recruiter configuration overrides automatic suggestions for essentiality."""
        config = {
            "essential_skills": ["Communication", "Leadership"],
            "optional_skills": ["Python"],
        }
        result = analyze_job_description(self.sample_jd, recruiter_config=config)

        # In sample_jd, Python was auto-essential and Communication was unspecified.
        # But with recruiter config, Communication is essential and Python is optional.
        self.assertListEqual(result["essential_skills"], ["Communication", "Leadership"])
        self.assertListEqual(result["optional_skills"], ["Python"])
        self.assertIn("Using recruiter-confirmed requirements", result["validation_messages"][0])


if __name__ == "__main__":
    unittest.main()
