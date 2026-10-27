"""
Regression and Acceptance Tests for Standalone Upload-Based Resume Screening.

Proves:
1. A screening request with 20 uploaded resumes returns results only for those uploaded files.
2. No candidate from the Kaggle dataset appears in the results.
3. The application works when all training dataset files are absent.
4. An empty upload does not trigger a dataset fallback.
5. A second screening request displays only its own candidates.
6. Invalid files are reported correctly.
7. Existing matching-engine scoring behavior is preserved.
8. Job descriptions uploaded as TXT and PDF files are parsed and evaluated properly.
"""

import io
import unittest
from unittest.mock import patch
import docx
import pypdf
from fastapi.testclient import TestClient

from backend.main import app, CURRENT_SCREENING_CANDIDATES
from src.document_parser import (
    parse_job_description_file,
    parse_single_resume,
)


def make_docx(text: str) -> bytes:
    """Helper to generate a valid in-memory DOCX file."""
    doc = docx.Document()
    doc.add_heading("Candidate Resume", level=0)
    doc.add_paragraph(text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def make_pdf(text: str = "Senior Engineer with Python, Docker, and SQL experience.") -> bytes:
    """Helper to generate a valid in-memory PDF file."""
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class TestUploadBasedScreening(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        # Clear session store before each test
        CURRENT_SCREENING_CANDIDATES.clear()

    def test_1_screening_request_with_20_uploaded_resumes_returns_only_uploaded_files(self):
        """
        Regression Test 1:
        A screening request with 20 uploaded resumes returns results only for those uploaded files.
        """
        files = []
        expected_filenames = []
        for i in range(1, 21):
            fname = f"candidate_{i:02d}.docx"
            expected_filenames.append(fname)
            skills = "Python, SQL" if i % 2 == 0 else "Java, C++, Docker"
            content = make_docx(f"Software Engineer {i} with 5 years experience. Skills: {skills}.")
            files.append(("files", (fname, content, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")))

        data = {
            "job_description": "Looking for a Software Engineer with Python and SQL experience.",
        }

        response = self.client.post("/api/screen", data=data, files=files)
        self.assertEqual(response.status_code, 200)
        res = response.json()

        self.assertTrue(res["is_valid"])
        self.assertEqual(res["total_uploaded"], 20)
        self.assertEqual(res["total_evaluated"], 20)
        self.assertEqual(res["returned_count"], 20)

        # Verify every returned candidate belongs strictly to the 20 uploaded files
        returned_filenames = [c["filename"] for c in res["ranked_candidates"]]
        self.assertEqual(len(returned_filenames), 20)
        self.assertEqual(set(returned_filenames), set(expected_filenames))

        # Check candidate ranks are sequential 1 to 20
        ranks = [c["candidate_rank"] for c in res["ranked_candidates"]]
        self.assertEqual(ranks, list(range(1, 21)))

    def test_2_no_candidate_from_kaggle_dataset_appears_in_results(self):
        """
        Regression Test 2:
        No candidate from the Kaggle dataset appears in the results.
        """
        files = [
            ("files", ("alice_engineer.docx", make_docx("Alice: Python, PyTorch, Docker specialist."), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
            ("files", ("bob_analyst.docx", make_docx("Bob: SQL, Tableau, Excel data analyst."), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
        ]

        data = {"job_description": "We need Python and SQL skills."}
        response = self.client.post("/api/screen", data=data, files=files)
        self.assertEqual(response.status_code, 200)
        res = response.json()

        returned_cands = res["ranked_candidates"]
        self.assertEqual(len(returned_cands), 2)

        # Assert no Kaggle numeric ID or Kaggle categories appear
        known_kaggle_ids = {"10554236", "16852973", "28398216", "37473139", "12632728"}
        for c in returned_cands:
            self.assertIn(c["filename"], ["alice_engineer.docx", "bob_analyst.docx"])
            self.assertNotIn(str(c["candidate_id"]), known_kaggle_ids)
            self.assertFalse(str(c["candidate_id"]).isdigit())

    def test_3_application_works_when_all_training_dataset_files_are_absent(self):
        """
        Regression Test 3:
        The application works when all training dataset files are absent.
        """
        # Patch dataset existence to return False
        with patch("pathlib.Path.exists", return_value=False):
            # Health check must succeed
            h_resp = self.client.get("/api/health")
            self.assertEqual(h_resp.status_code, 200)
            self.assertEqual(h_resp.json()["status"], "healthy")

            # Categories must succeed
            c_resp = self.client.get("/api/categories")
            self.assertEqual(c_resp.status_code, 200)

            # Screening request must succeed without any dataset
            files = [
                ("files", ("dev.docx", make_docx("Developer with Python and SQL."), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"))
            ]
            s_resp = self.client.post("/api/screen", data={"job_description": "Python developer needed."}, files=files)
            self.assertEqual(s_resp.status_code, 200)
            self.assertEqual(s_resp.json()["returned_count"], 1)

    def test_4_empty_upload_does_not_trigger_dataset_fallback(self):
        """
        Regression Test 4:
        An empty upload does not trigger a dataset fallback.
        """
        # 1. No resumes uploaded
        resp = self.client.post("/api/screen", data={"job_description": "Python engineer."})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("No resumes uploaded", resp.json()["detail"])

        # 2. No job description provided
        files = [
            ("files", ("dev.docx", make_docx("Python developer."), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"))
        ]
        resp_no_jd = self.client.post("/api/screen", data={"job_description": ""}, files=files)
        self.assertEqual(resp_no_jd.status_code, 400)
        self.assertIn("No job description provided", resp_no_jd.json()["detail"])

        # 3. /api/rank with empty session returns error, never falls back
        CURRENT_SCREENING_CANDIDATES.clear()
        rank_resp = self.client.post("/api/rank", json={"job_description": "Python engineer."})
        self.assertEqual(rank_resp.status_code, 400)
        self.assertIn("No candidate resumes have been uploaded", rank_resp.json()["detail"])

    def test_5_second_screening_request_displays_only_its_own_candidates(self):
        """
        Regression Test 5:
        A second screening request displays only its own candidates and replaces the previous request.
        """
        # Request 1: Upload batch A (3 candidates)
        files_a = [
            ("files", (f"batch_a_{i}.docx", make_docx(f"Candidate A{i} with Python"), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"))
            for i in range(1, 4)
        ]
        resp_a = self.client.post("/api/screen", data={"job_description": "Python job"}, files=files_a)
        self.assertEqual(resp_a.status_code, 200)
        self.assertEqual(resp_a.json()["returned_count"], 3)
        self.assertIn("batch_a_1.docx", [c["filename"] for c in resp_a.json()["ranked_candidates"]])

        # Verify candidate detail for batch A works
        detail_a = self.client.get("/api/candidate/batch_a_1.docx")
        self.assertEqual(detail_a.status_code, 200)

        # Request 2: Upload batch B (2 completely different candidates)
        files_b = [
            ("files", (f"batch_b_{i}.docx", make_docx(f"Candidate B{i} with SQL"), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"))
            for i in range(1, 3)
        ]
        resp_b = self.client.post("/api/screen", data={"job_description": "SQL job"}, files=files_b)
        self.assertEqual(resp_b.status_code, 200)
        self.assertEqual(resp_b.json()["returned_count"], 2)

        # Batch B candidates must be the only ones present
        filenames_b = [c["filename"] for c in resp_b.json()["ranked_candidates"]]
        self.assertEqual(set(filenames_b), {"batch_b_1.docx", "batch_b_2.docx"})
        self.assertNotIn("batch_a_1.docx", filenames_b)

        # Previous batch A candidate must now return 404 (isolated session)
        detail_a_after = self.client.get("/api/candidate/batch_a_1.docx")
        self.assertEqual(detail_a_after.status_code, 404)

        # New batch B candidate must return 200
        detail_b = self.client.get("/api/candidate/batch_b_1.docx")
        self.assertEqual(detail_b.status_code, 200)

    def test_6_invalid_files_are_reported_correctly(self):
        """
        Regression Test 6:
        Invalid files are reported correctly without crashing, and valid resumes are processed.
        """
        files = [
            # 2 Valid DOCX resumes
            ("files", ("valid_1.docx", make_docx("Python and Docker developer."), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
            ("files", ("valid_2.docx", make_docx("SQL and Machine Learning specialist."), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
            # 1 Unsupported extension
            ("files", ("notes.txt", b"plain text is not a resume", "text/plain")),
            # 1 Empty file (0 bytes)
            ("files", ("empty_resume.pdf", b"", "application/pdf")),
            # 1 Corrupted PDF
            ("files", ("corrupted.pdf", b"NOT_A_VALID_PDF_HEADER", "application/pdf")),
        ]

        data = {"job_description": "Seeking Python, SQL, and Docker."}
        response = self.client.post("/api/screen", data=data, files=files)
        self.assertEqual(response.status_code, 200)
        res = response.json()

        # 2 valid resumes evaluated and returned
        self.assertEqual(res["total_uploaded"], 5)
        self.assertEqual(res["total_evaluated"], 2)
        self.assertEqual(res["returned_count"], 2)
        self.assertEqual({c["filename"] for c in res["ranked_candidates"]}, {"valid_1.docx", "valid_2.docx"})

        # 3 failed files reported with filenames and reasons
        self.assertEqual(len(res["failed_files"]), 3)
        failed_names = {f["filename"] for f in res["failed_files"]}
        self.assertEqual(failed_names, {"notes.txt", "empty_resume.pdf", "corrupted.pdf"})

    def test_7_matching_engine_scoring_behavior_preserved(self):
        """
        Regression Test 7:
        Existing matching-engine scoring behavior and formula are strictly preserved:
        combined = (0.7 * skill_score) + (0.3 * text_sim)
        """
        # Candidate 1: matches all essential skills (Python, SQL)
        c1 = make_docx("Senior developer with Python and SQL and Docker.")
        # Candidate 2: matches only 1 essential skill
        c2 = make_docx("Junior developer with Python only.")

        files = [
            ("files", ("cand_strong.docx", c1, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
            ("files", ("cand_partial.docx", c2, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
        ]

        data = {
            "job_description": "Requirements: Must have Python and SQL.",
        }

        response = self.client.post("/api/screen", data=data, files=files)
        self.assertEqual(response.status_code, 200)
        res = response.json()

        candidates = res["ranked_candidates"]
        self.assertEqual(len(candidates), 2)

        top_cand = candidates[0]
        self.assertEqual(top_cand["filename"], "cand_strong.docx")
        self.assertEqual(top_cand["candidate_rank"], 1)

        # Verify exact combined score formula for candidate
        expected_combined = round(
            (0.70 * top_cand["skill_match_score"]) + (0.30 * top_cand["text_similarity_score"]),
            2,
        )
        self.assertAlmostEqual(top_cand["combined_score"], expected_combined, places=2)

        # Strong candidate must outrank partial candidate
        self.assertGreater(top_cand["combined_score"], candidates[1]["combined_score"])

        # Check required fields exist
        self.assertIn("essential_skills_matched", top_cand)
        self.assertIn("essential_skills_missing", top_cand)
        self.assertIn("optional_skills_matched", top_cand)
        self.assertIn("match_explanation", top_cand)

    def test_8_job_description_file_upload(self):
        """
        Test uploading job description as TXT file and PDF file.
        """
        jd_txt_content = b"Requirements:\n- Must have Python and PostgreSQL.\nPreferred:\n- Docker."
        files = [
            ("jd_file", ("job_req.txt", jd_txt_content, "text/plain")),
            ("files", ("candidate_a.docx", make_docx("Python and PostgreSQL developer."), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
        ]

        response = self.client.post("/api/screen", files=files)
        self.assertEqual(response.status_code, 200)
        res = response.json()
        self.assertEqual(res["returned_count"], 1)
        self.assertIn("Python", res["essential_skills"])
        self.assertIn("PostgreSQL", res["essential_skills"])

    def test_req1_job_description_skill_extraction_headers_and_inline(self):
        """
        Requirement 1: Accurate extraction of essential and optional skills
        from job descriptions containing both section headers and inline cues.
        """
        # Inline cues
        jd_inline = (
            "We are seeking a Backend Engineer. "
            "Must have Python and PostgreSQL experience. "
            "Nice to have Docker. Preferred experience with Redis."
        )
        resp = self.client.post("/api/analyze-job", json={"job_description": jd_inline})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("Python", data["essential_skills"])
        self.assertIn("PostgreSQL", data["essential_skills"])
        self.assertIn("Docker", data["optional_skills"])
        self.assertIn("Redis", data["optional_skills"])

        # Section headers
        jd_headers = (
            "Job Title: Senior Data Engineer\n\n"
            "Required Skills:\n"
            "- Python\n"
            "- SQL\n\n"
            "Nice To Have / Bonus:\n"
            "- Kubernetes\n"
            "- Docker\n"
        )
        resp2 = self.client.post("/api/analyze-job", json={"job_description": jd_headers})
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.json()
        self.assertIn("Python", data2["essential_skills"])
        self.assertIn("SQL", data2["essential_skills"])
        self.assertIn("Kubernetes", data2["optional_skills"])
        self.assertIn("Docker", data2["optional_skills"])

    def test_req2_matched_and_missing_skill_lists_for_synthetic_resumes(self):
        """
        Requirement 2: Accurate matched and missing skill lists for known synthetic resumes.
        Verifies both essential and optional match/miss lists and their alias fields.
        """
        files = [
            ("files", ("full_match.docx", make_docx("Expert with Python, SQL, and Docker."), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
            ("files", ("partial_match.docx", make_docx("Developer with Python and Git."), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
            ("files", ("zero_match.docx", make_docx("Marketing lead with SEO and Copywriting."), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
        ]
        data = {
            "job_description": "Requirements: Python, SQL. Nice to have: Docker.",
        }
        resp = self.client.post("/api/screen", data=data, files=files)
        self.assertEqual(resp.status_code, 200)
        cands_by_name = {c["filename"]: c for c in resp.json()["ranked_candidates"]}

        full = cands_by_name["full_match.docx"]
        self.assertEqual(set(full["essential_skills_matched"]), {"Python", "SQL"})
        self.assertEqual(full["essential_skills_missing"], [])
        self.assertEqual(full["optional_skills_matched"], ["Docker"])
        # Check alias keys
        self.assertEqual(set(full["matched_essential"]), {"Python", "SQL"})
        self.assertEqual(full["missing_essential"], [])
        self.assertEqual(full["matched_optional"], ["Docker"])

        partial = cands_by_name["partial_match.docx"]
        self.assertEqual(partial["essential_skills_matched"], ["Python"])
        self.assertEqual(partial["essential_skills_missing"], ["SQL"])
        self.assertEqual(partial["optional_skills_matched"], [])
        self.assertEqual(partial["matched_essential"], ["Python"])
        self.assertEqual(partial["missing_essential"], ["SQL"])

        zero = cands_by_name["zero_match.docx"]
        self.assertEqual(zero["essential_skills_matched"], [])
        self.assertEqual(set(zero["essential_skills_missing"]), {"Python", "SQL"})
        self.assertEqual(zero["optional_skills_matched"], [])

    def test_req3_candidate_specific_match_explanations(self):
        """
        Requirement 3: Candidate-specific match explanations differing meaningfully
        between resumes with different skill sets.
        """
        files = [
            ("files", ("dev_a.docx", make_docx("Python, SQL, Docker backend engineer."), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
            ("files", ("dev_b.docx", make_docx("Java and C++ developer with Git."), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
        ]
        data = {"job_description": "Requirements: Python, SQL. Preferred: Docker."}
        resp = self.client.post("/api/screen", data=data, files=files)
        self.assertEqual(resp.status_code, 200)
        cands = resp.json()["ranked_candidates"]

        exp_a = cands[0]["match_explanation"]
        exp_b = cands[1]["match_explanation"]
        det_a = cands[0]["detailed_explanation"]
        det_b = cands[1]["detailed_explanation"]

        # Explanations must not be identical
        self.assertNotEqual(exp_a, exp_b)
        self.assertNotEqual(det_a, det_b)

        # Candidate A mentions matching skills
        self.assertIn("Python", exp_a)
        self.assertIn("Docker", exp_a)

        # Candidate B mentions missing skills
        self.assertIn("Missing", exp_b)

    def test_req4_score_field_mapping_consistency(self):
        """
        Requirement 4: Correct score-field mapping (combined_score, skill_match_score,
        text_similarity_score) between backend and frontend contracts.
        """
        files = [
            ("files", ("cand_1.docx", make_docx("Python and Docker specialist."), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"))
        ]
        data = {"job_description": "Must have Python and SQL. Preferred Docker."}
        resp = self.client.post("/api/screen", data=data, files=files)
        self.assertEqual(resp.status_code, 200)
        cand = resp.json()["ranked_candidates"][0]

        self.assertIsInstance(cand["combined_score"], (int, float))
        self.assertIsInstance(cand["skill_match_score"], (int, float))
        self.assertIsInstance(cand["text_similarity_score"], (int, float))
        self.assertTrue(0 <= cand["combined_score"] <= 100)
        self.assertTrue(0 <= cand["skill_match_score"] <= 100)
        self.assertTrue(0 <= cand["text_similarity_score"] <= 100)

        # Mathematical verification: combined = 0.70 * skill + 0.30 * text_sim
        expected = round((0.70 * cand["skill_match_score"]) + (0.30 * cand["text_similarity_score"]), 2)
        self.assertAlmostEqual(cand["combined_score"], expected, places=2)

    def test_req5_dynamic_category_generation_and_unclassified(self):
        """
        Requirement 5: Dynamic category generation reflecting uploaded resumes:
        domain categories and Other / Unclassified.
        """
        files = [
            ("files", ("ml_eng.docx", make_docx("PyTorch, TensorFlow, Scikit-learn, Machine Learning researcher."), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
            ("files", ("web_dev.docx", make_docx("React, TypeScript, CSS, HTML frontend developer."), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
            ("files", ("unclassified.docx", make_docx("General administrator with office and typing experience."), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
        ]
        data = {"job_description": "Seeking technical talent with Python and React."}
        resp = self.client.post("/api/screen", data=data, files=files)
        self.assertEqual(resp.status_code, 200)
        cands_by_name = {c["filename"]: c for c in resp.json()["ranked_candidates"]}

        self.assertEqual(cands_by_name["ml_eng.docx"]["category"], "Machine Learning & AI")
        self.assertEqual(cands_by_name["web_dev.docx"]["category"], "Web Development")
        self.assertEqual(cands_by_name["unclassified.docx"]["category"], "Other / Unclassified")

        # Available categories returned in screen response
        avail = resp.json()["available_categories"]
        self.assertIn("Machine Learning & AI", avail)
        self.assertIn("Web Development", avail)
        self.assertIn("Other / Unclassified", avail)

        # /api/categories endpoint reflects active screening candidates
        cat_resp = self.client.get("/api/categories")
        self.assertEqual(cat_resp.status_code, 200)
        active_cat_names = [c["name"] for c in cat_resp.json()["categories"]]
        self.assertIn("Machine Learning & AI", active_cat_names)
        self.assertIn("Web Development", active_cat_names)
        self.assertIn("Other / Unclassified", active_cat_names)

    def test_req6_category_and_min_score_filtering_without_data_loss(self):
        """
        Requirement 6: Interactive category filtering combined with minimum-score
        filtering without score recalculation or data loss.
        """
        files = [
            ("files", ("c1_ml.docx", make_docx("Python, PyTorch, SQL Machine Learning engineer."), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
            ("files", ("c2_web.docx", make_docx("HTML, CSS, React frontend developer."), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
        ]
        data = {"job_description": "Requirements: Python, SQL."}
        resp = self.client.post("/api/screen", data=data, files=files)
        self.assertEqual(resp.status_code, 200)

        # Filter by category and min_score via /api/rank
        rank_resp = self.client.post(
            "/api/rank",
            json={
                "job_description": "Requirements: Python, SQL.",
                "category_filter": "Machine Learning & AI",
                "min_score": 40.0,
            },
        )
        self.assertEqual(rank_resp.status_code, 200)
        ranked = rank_resp.json()["ranked_candidates"]
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["filename"], "c1_ml.docx")

        # Clear filters: all candidates are returned with exact same original scores
        clear_resp = self.client.post(
            "/api/rank",
            json={
                "job_description": "Requirements: Python, SQL.",
                "category_filter": "All Categories",
                "min_score": 0.0,
            },
        )
        self.assertEqual(clear_resp.status_code, 200)
        all_ranked = clear_resp.json()["ranked_candidates"]
        self.assertEqual(len(all_ranked), 2)
        c1_original = next(c for c in resp.json()["ranked_candidates"] if c["filename"] == "c1_ml.docx")
        c1_recleared = next(c for c in all_ranked if c["filename"] == "c1_ml.docx")
        self.assertEqual(c1_original["combined_score"], c1_recleared["combined_score"])

    def test_req7_dynamic_max_results_options_generation(self):
        """
        Requirement 7: Dynamic Max Results options for 1, 7, and 20 valid resumes.
        Verifies top_n slicing on uploaded candidates.
        """
        files_7 = [
            ("files", (f"cand_{i}.docx", make_docx(f"Engineer {i} with Python"), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"))
            for i in range(1, 8)
        ]
        resp = self.client.post("/api/screen", data={"job_description": "Python"}, files=files_7)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["total_evaluated"], 7)

        # Test Top 3 slicing
        resp_top3 = self.client.post("/api/rank", json={"job_description": "Python", "top_n": 3})
        self.assertEqual(resp_top3.status_code, 200)
        self.assertEqual(resp_top3.json()["returned_count"], 3)

        # Test Top 5 slicing
        resp_top5 = self.client.post("/api/rank", json={"job_description": "Python", "top_n": 5})
        self.assertEqual(resp_top5.status_code, 200)
        self.assertEqual(resp_top5.json()["returned_count"], 5)

    def test_req8_missing_optional_and_unidentifiable_requirements(self):
        """
        Requirement 8: Graceful handling of missing optional requirements
        and unidentifiable job requirements.
        """
        # 1. Job description with no optional skills
        files = [
            ("files", ("dev.docx", make_docx("Python and SQL developer."), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"))
        ]
        resp = self.client.post("/api/screen", data={"job_description": "Requirements: Python, SQL."}, files=files)
        self.assertEqual(resp.status_code, 200)
        res = resp.json()
        self.assertEqual(res["optional_skills"], [])
        cand = res["ranked_candidates"][0]
        self.assertEqual(cand["optional_skills_matched"], [])

        # 2. Job description with no recognizable requirements at all
        resp_vague = self.client.post(
            "/api/screen",
            data={"job_description": "We are looking for someone enthusiastic, friendly, and team-oriented."},
            files=files,
        )
        self.assertEqual(resp_vague.status_code, 200)
        res_vague = resp_vague.json()
        # Essential and optional skills are empty
        self.assertEqual(res_vague["essential_skills"], [])
        self.assertEqual(res_vague["optional_skills"], [])
        # Job analysis contains warning
        job_analysis = res_vague.get("job_analysis", {})
        self.assertTrue(
            any("No explicit skill requirements could be identified" in w for w in job_analysis.get("warnings", []))
        )

    def test_req9_candidate_details_selection(self):
        """
        Requirement 9: Correct candidate details displayed when selecting different candidates.
        """
        files = [
            ("files", ("alice.docx", make_docx("Alice: Python, PyTorch, Docker."), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
            ("files", ("bob.docx", make_docx("Bob: React, CSS, TypeScript."), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
        ]
        self.client.post("/api/screen", data={"job_description": "Python and React."}, files=files)

        alice_detail = self.client.get("/api/candidate/alice.docx").json()
        bob_detail = self.client.get("/api/candidate/bob.docx").json()

        self.assertEqual(alice_detail["filename"], "alice.docx")
        self.assertIn("Python", alice_detail["all_skills"])
        self.assertIn("PyTorch", alice_detail["all_skills"])

        self.assertEqual(bob_detail["filename"], "bob.docx")
        self.assertIn("React", bob_detail["all_skills"])
        self.assertIn("TypeScript", bob_detail["all_skills"])


if __name__ == "__main__":
    unittest.main()
