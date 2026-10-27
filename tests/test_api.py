"""
Unit and integration tests for FastAPI backend API endpoints.

Tests:
- GET /api/health
- GET /api/categories
- POST /api/analyze-job (valid and empty input)
- POST /api/screen (standalone upload screening)
- POST /api/rank (session candidate ranking)
- GET /api/candidate/{id} (valid ID and non-existent ID)
- POST /api/upload-resumes
"""

import io
import unittest
import docx
from fastapi.testclient import TestClient
from backend.main import app, CURRENT_SCREENING_CANDIDATES


def make_docx(text: str) -> bytes:
    doc = docx.Document()
    doc.add_heading("Jane Doe", 0)
    doc.add_paragraph(text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


class TestBackendAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        CURRENT_SCREENING_CANDIDATES.clear()

    def test_health_endpoint(self):
        """Test GET /api/health returns 200 and healthy status."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["mode"], "standalone_upload")
        self.assertIn("default_scoring_weights", data)

    def test_categories_endpoint(self):
        """Test GET /api/categories returns skill taxonomy categories."""
        response = self.client.get("/api/categories")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreater(data["total_categories"], 0)
        cats = [c["name"] for c in data["categories"]]
        self.assertIn("Programming Languages", cats)
        self.assertIn("Machine Learning & AI", cats)

    def test_analyze_job_valid(self):
        """Test POST /api/analyze-job with valid description."""
        payload = {
            "job_description": "We need a Senior Python Developer with SQL and Docker experience.",
        }
        response = self.client.post("/api/analyze-job", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["is_valid"])
        self.assertIn("Python", data["extracted_skills"])
        self.assertIn("SQL", data["extracted_skills"])
        self.assertIn("Docker", data["extracted_skills"])

    def test_analyze_job_empty(self):
        """Test POST /api/analyze-job with empty description returns 400."""
        payload = {"job_description": "   \n\t  "}
        response = self.client.post("/api/analyze-job", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("detail", response.json())

    def test_screen_endpoint_valid(self):
        """Test POST /api/screen returns ranked candidates for uploaded files."""
        files = [
            ("files", ("cand1.docx", make_docx("Python, SQL, and Docker engineer."), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
            ("files", ("cand2.docx", make_docx("JavaScript and CSS web developer."), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
        ]
        data = {
            "job_description": "Requirements:\n- Must have Python, SQL, and Docker.",
        }
        response = self.client.post("/api/screen", data=data, files=files)
        self.assertEqual(response.status_code, 200)
        res = response.json()
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["total_evaluated"], 2)
        self.assertEqual(res["returned_count"], 2)

        cands = res["ranked_candidates"]
        self.assertEqual(cands[0]["filename"], "cand1.docx")
        self.assertGreater(cands[0]["combined_score"], cands[1]["combined_score"])
        self.assertIn("Python", cands[0]["essential_skills_matched"])

    def test_rank_candidates_empty_session_returns_400(self):
        """Test POST /api/rank without uploaded resumes returns 400."""
        payload = {"job_description": "Python developer"}
        response = self.client.post("/api/rank", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("No candidate resumes have been uploaded", response.json()["detail"])

    def test_rank_candidates_empty_description(self):
        """Test POST /api/rank with empty description returns 400."""
        payload = {"job_description": ""}
        response = self.client.post("/api/rank", json=payload)
        self.assertEqual(response.status_code, 400)

    def test_get_candidate_details_not_found(self):
        """Test GET /api/candidate/{id} for non-existent ID returns 404."""
        response = self.client.get("/api/candidate/non_existent.docx")
        self.assertEqual(response.status_code, 404)

    def test_upload_resumes_valid_docx(self):
        """Test POST /api/upload-resumes with a valid DOCX file."""
        files = [
            ("files", ("jane_resume.docx", make_docx("Machine learning engineer with Python, PyTorch, and SQL."), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"))
        ]
        response = self.client.post("/api/upload-resumes", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_files"], 1)
        self.assertEqual(data["valid_count"], 1)
        self.assertEqual(data["error_count"], 0)
        self.assertTrue(data["files"][0]["is_valid"])
        self.assertIn("Python", data["files"][0]["extracted_skills"])


if __name__ == "__main__":
    unittest.main()
