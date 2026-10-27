"""
Regression Tests for Resume Screening Across Three Different Job Descriptions.

Validates:
1. Junior Machine Learning Engineer:
   - resume_01.pdf and resume_06.pdf legitimately rank #1 and #2 due to strong ML/DL skills.
2. Data Analyst / Junior Data Scientist:
   - resume_07.pdf and resume_08.pdf legitimately rank #1 and #2 (Excel, Power BI, SQL, Tableau, etc.).
   - resume_01.pdf and resume_06.pdf do NOT rank #1 and #2 (they lack Excel, Power BI, Tableau).
3. Python Backend Developer Intern:
   - resume_15.pdf and resume_13.pdf legitimately rank #1 and #2 (FastAPI/Django, PostgreSQL, Docker, Redis).
   - resume_01.pdf does NOT rank #1 or #2.
4. Consecutive screening requests recalculate all scores, ranks, and explanations without stale state carryover.
5. GET /api/candidate/{id} returns updated scores and match explanations for the currently evaluated job.
"""

import unittest
from pathlib import Path
from fastapi.testclient import TestClient

from backend.main import app, CURRENT_SCREENING_CANDIDATES

JD_ML = """Job Title: Junior Machine Learning Engineer / AI Engineer Intern

Role Overview
We are looking for a motivated Junior Machine Learning Engineer / AI Engineer Intern to join our applied AI team. You will work alongside senior engineers to prepare data, train and evaluate machine learning models, and help move prototypes toward production. This is a hands-on role for someone who enjoys experimenting with data and models and learning how ML systems are built.

Responsibilities
- Clean, transform and explore datasets for model training using Python, NumPy and Pandas.
- Build, train and evaluate supervised and unsupervised machine learning models.
- Run experiments, compare model performance using appropriate metrics, and document findings.
- Perform feature engineering and hyperparameter tuning to improve model quality.
- Support the team in packaging models and exposing them for internal use.
- Write clean, readable, version-controlled code and take part in code reviews.

Essential Skills (Required)
- Python programming
- NumPy and Pandas for data manipulation
- scikit-learn for building and evaluating models
- Machine learning fundamentals: supervised and unsupervised learning, model evaluation, cross-validation
- Git and GitHub for version control

Optional Skills (Preferred / Nice to Have)
- Deep learning frameworks such as PyTorch or TensorFlow
- Natural Language Processing (NLP) or Computer Vision experience
- Hugging Face Transformers
- Docker
- FastAPI or Flask for serving models
- MLflow or other experiment tracking tools
- Basic SQL
- Familiarity with cloud platforms such as AWS
- Jupyter Notebook

Education and Experience
- Currently pursuing or recently completed a Bachelor's or Master's degree in Computer Science, Artificial Intelligence, Data Science, Information Technology or a related field.
- Prior internship experience is not required, but academic or personal machine learning projects are expected.
- Strong analytical thinking and willingness to learn.
"""

JD_DA = """Job Title: Data Analyst / Junior Data Scientist

Role Overview
We are hiring a Data Analyst / Junior Data Scientist to help business teams make better decisions using data. You will query databases, clean and analyse data, build dashboards and present insights clearly to non-technical stakeholders. This position is suitable for recent graduates and interns with a strong foundation in data analysis.

Responsibilities
- Write SQL queries to extract and aggregate data from relational databases.
- Clean and prepare raw data for analysis using Python, Pandas and Excel.
- Build interactive dashboards and reports with Power BI or Tableau.
- Apply statistical methods to analyse trends, test hypotheses and measure performance.
- Summarise findings in clear visualisations and short written reports.
- Collaborate with product, marketing and operations teams to define metrics.

Essential Skills (Required)
- SQL for querying and aggregating data
- Microsoft Excel (pivot tables, formulas, lookups)
- Python with Pandas for data analysis
- Data visualization using Power BI or Tableau
- Statistics fundamentals (descriptive statistics, hypothesis testing)
- Data cleaning and preparation

Optional Skills (Preferred / Nice to Have)
- Matplotlib or Seaborn
- scikit-learn for basic predictive modelling
- A/B testing
- Google Analytics
- R programming
- Jupyter Notebook
- Experience with BigQuery or other cloud data warehouses

Education and Experience
- Bachelor's or Master's degree (completed or in progress) in Statistics, Mathematics, Computer Science, Economics, Data Science, Business Analytics or a related field.
- Internship or project experience in data analysis is a plus but not mandatory.
- Good communication skills and attention to detail.
"""

JD_BACKEND = """Job Title: Python Backend Developer Intern

Role Overview
We are looking for a Python Backend Developer Intern to join our engineering team. You will help design, build and test server-side features and REST APIs that power our web and mobile products. You will learn how production backend systems are built, tested and deployed while working with experienced engineers.

Responsibilities
- Develop and maintain backend services and REST APIs using Python.
- Design database schemas and write efficient SQL queries.
- Implement user authentication and authorisation for API endpoints.
- Write unit and integration tests for new features.
- Debug issues, improve performance and contribute to documentation.
- Use Git workflows, pull requests and code reviews as part of daily development.

Essential Skills (Required)
- Python programming
- Django, Flask or FastAPI web framework
- REST API design and development
- SQL with a relational database such as PostgreSQL or MySQL
- Git and GitHub
- Authentication basics (JWT, sessions or OAuth)

Optional Skills (Preferred / Nice to Have)
- Docker
- Redis or other caching tools
- Celery or background task queues
- AWS or another cloud platform
- CI/CD pipelines (for example GitHub Actions)
- Automated testing with pytest
- Linux command line
- MongoDB or other NoSQL databases

Education and Experience
- Currently pursuing or recently completed a Bachelor's degree in Computer Science, Information Technology, Software Engineering or a related field.
- Personal or academic backend projects are expected; prior internship experience is a plus.
- Ability to work in a team and learn quickly.
"""


class TestThreeJobsScreeningRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.resumes_dir = Path(__file__).resolve().parent.parent / "data" / "test_resumes"
        cls.resume_files = []
        for i in range(1, 21):
            fname = f"resume_{i:02d}.pdf"
            fpath = cls.resumes_dir / fname
            with open(fpath, "rb") as f:
                content = f.read()
            cls.resume_files.append((fname, content))

    def setUp(self):
        CURRENT_SCREENING_CANDIDATES.clear()

    def get_multipart_files(self):
        return [("files", (fname, content, "application/pdf")) for fname, content in self.resume_files]

    def test_job_1_machine_learning_rankings(self):
        """Test that resume_01.pdf and resume_06.pdf legitimately top ML Engineer role."""
        resp = self.client.post("/api/screen", data={"job_description": JD_ML}, files=self.get_multipart_files())
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["total_evaluated"], 20)
        cands = data["ranked_candidates"]

        # Resume 01 is #1
        self.assertEqual(cands[0]["filename"], "resume_01.pdf")
        self.assertEqual(cands[0]["candidate_rank"], 1)
        self.assertGreaterEqual(cands[0]["combined_score"], 70.0)
        self.assertIn("Machine Learning", cands[0]["matched_essential"])
        self.assertIn("Python", cands[0]["matched_essential"])

        # Resume 06 is #2
        self.assertEqual(cands[1]["filename"], "resume_06.pdf")
        self.assertEqual(cands[1]["candidate_rank"], 2)

    def test_job_2_data_analyst_rankings(self):
        """Test that Data Analyst role ranks analytical profiles first, NOT resume_01 or resume_06."""
        resp = self.client.post("/api/screen", data={"job_description": JD_DA}, files=self.get_multipart_files())
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        cands = data["ranked_candidates"]
        self.assertEqual(cands[0]["filename"], "resume_07.pdf")
        self.assertEqual(cands[1]["filename"], "resume_08.pdf")

        # resume_01 and resume_06 must NOT be in top 2
        top_2_names = [cands[0]["filename"], cands[1]["filename"]]
        self.assertNotIn("resume_01.pdf", top_2_names)
        self.assertNotIn("resume_06.pdf", top_2_names)

        # Verify missing skills for resume_01
        cand_01 = next(c for c in cands if c["filename"] == "resume_01.pdf")
        self.assertIn("Excel", cand_01["missing_essential"])
        self.assertIn("Tableau", cand_01["missing_essential"])
        self.assertGreater(cand_01["candidate_rank"], 5)

    def test_job_3_backend_developer_rankings(self):
        """Test that Backend Developer role ranks backend profiles first, NOT resume_01."""
        resp = self.client.post("/api/screen", data={"job_description": JD_BACKEND}, files=self.get_multipart_files())
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        cands = data["ranked_candidates"]
        self.assertEqual(cands[0]["filename"], "resume_15.pdf")
        self.assertEqual(cands[1]["filename"], "resume_13.pdf")

        # resume_01 must NOT be in top 2
        top_2_names = [cands[0]["filename"], cands[1]["filename"]]
        self.assertNotIn("resume_01.pdf", top_2_names)

        # Verify missing essential skills for resume_01 on Backend role
        cand_01 = next(c for c in cands if c["filename"] == "resume_01.pdf")
        self.assertIn("FastAPI", cand_01["missing_essential"])
        self.assertIn("Django", cand_01["missing_essential"])

    def test_consecutive_screenings_do_not_carry_stale_results(self):
        """
        Verify that screening Job 1, then Job 2, then Job 3 successively
        recalculates all candidate rankings, scores, and candidate details cleanly.
        """
        # 1. Screen Job 1
        resp1 = self.client.post("/api/screen", data={"job_description": JD_ML}, files=self.get_multipart_files())
        self.assertEqual(resp1.status_code, 200)
        cands1 = resp1.json()["ranked_candidates"]
        self.assertEqual(cands1[0]["filename"], "resume_01.pdf")
        detail1 = self.client.get("/api/candidate/resume_01.pdf").json()
        self.assertEqual(detail1["combined_score"], cands1[0]["combined_score"])
        self.assertIn("Machine Learning", detail1["matched_essential"])

        # 2. Screen Job 2 with fresh JD
        resp2 = self.client.post("/api/screen", data={"job_description": JD_DA}, files=self.get_multipart_files())
        self.assertEqual(resp2.status_code, 200)
        cands2 = resp2.json()["ranked_candidates"]
        self.assertEqual(cands2[0]["filename"], "resume_07.pdf")
        self.assertNotEqual(cands2[0]["filename"], "resume_01.pdf")

        # Candidate detail for resume_01 must now reflect Data Analyst requirements (missing Excel/Tableau)
        detail2 = self.client.get("/api/candidate/resume_01.pdf").json()
        self.assertEqual(detail2["combined_score"], next(c["combined_score"] for c in cands2 if c["filename"] == "resume_01.pdf"))
        self.assertIn("Excel", detail2["missing_essential"])

        # 3. Screen Job 3 with fresh JD
        resp3 = self.client.post("/api/screen", data={"job_description": JD_BACKEND}, files=self.get_multipart_files())
        self.assertEqual(resp3.status_code, 200)
        cands3 = resp3.json()["ranked_candidates"]
        self.assertEqual(cands3[0]["filename"], "resume_15.pdf")
        self.assertNotEqual(cands3[0]["filename"], "resume_01.pdf")


if __name__ == "__main__":
    unittest.main()
