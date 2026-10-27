# AI-Powered Resume Screening & Candidate Ranking System

**Internship Program:** Future Interns Machine Learning  
**Task:** Task 3 — Resume Screening and Candidate Ranking System  
**Current Status:** Stage 6 — Frontend Web Application Completed  

---

## 📌 Project Overview
In modern recruitment processes, human resource teams receive hundreds or thousands of resumes for a single job opening. Manually screening each resume is time-consuming, prone to cognitive fatigue, and subjective. 

This project builds an end-to-end **Resume Screening and Candidate Ranking System** leveraging Natural Language Processing (NLP) and Machine Learning (ML) techniques. The system automatically ingests resumes, extracts skills with evidence snippets, parses Job Descriptions (JDs), and calculates explainable candidate match scores using a hybrid of **TF-IDF textual cosine similarity** and **weighted skill coverage**.

---

## 🎯 Key Objectives
1. **Automated Resume Parsing & Cleaning:** Ingest resumes, strip boilerplate and noise, preserve technical terms (e.g., C++, C#, .NET, Node.js), and standardize text representations.
2. **Explainable Skill Extraction:** Detect technical and professional skills using transparent, rule-based NLP with evidence snippets.
3. **Job Description Analysis:** Parse job descriptions, distinguish essential vs. optional skills, and allow recruiter customization with unknown skill detection.
4. **Candidate Matching & Ranking Engine:** Compute explainable scores combining TF-IDF cosine similarity and weighted skill coverage with deterministic ranking.
5. **Interactive Web Dashboard:** Provide a clean, user-friendly HTML/CSS/JS web interface backed by a FastAPI REST API for recruiters to paste job descriptions, review skills, rank candidates, and inspect match analytics.

---

## 📊 Dataset Overview
The project utilizes the Kaggle Resume Dataset stored in the `archive/` directory:
- **Primary Dataset File:** `archive/Resume/Resume.csv`
- **Associated PDF Resumes:** `archive/data/data/<CATEGORY>/<ID>.pdf`
- **Total Records:** 2,484 resumes
- **Total Categories:** 24 balanced professional domains
- **Original Columns:**
  - `ID`: Unique identifier for each resume.
  - `Resume_str`: Plain-text extract of the resume content.
  - `Resume_html`: Formatted HTML representation of the resume.
  - `Category`: Target job domain (e.g., `INFORMATION-TECHNOLOGY`, `HR`, `FINANCE`, `ADVOCATE`, `ENGINEERING`, etc.).

---

## 🧹 Stage 2: Resume Text Processing & Preprocessing
The preprocessing module ([`src/text_preprocessor.py`](src/text_preprocessor.py)) implements a clean, modular, and non-destructive pipeline:
- **Technical Term Preservation:** Explicitly protects symbols in `C++`, `C#`, `.NET`, `Node.js`, `PL/SQL`, `CI/CD`.
- **Missing & Empty Resume Resolution:** Identified row index 656 (`ID: 12632728`) as completely empty in text, HTML, and PDF; safely preserved and flagged as `is_usable = False`.
- **Non-Destructive Deduplication:** Identified 2 pairs of duplicate resumes (`ID: 28398216` and `ID: 37473139`); flagged as `is_duplicate = True` without deleting raw records.
- **Processed Dataset Output:** Saved to `data/processed/resumes_cleaned.csv` (28.65 MB, 2,484 rows).

---

## 🔍 Stage 3: NLP-Based Skill Extraction
The skill extraction module ([`src/skill_extractor.py`](src/skill_extractor.py)) provides an explainable, rule-based skill extraction engine:
- **Taxonomy:** Curated dictionary across 6 domains (Programming Languages, Machine Learning & AI, Data Tools, Web Development, Databases & Dev Tools, Professional Skills).
- **Context Guards:** Eliminates false positives on single-letter languages (`C`, `R`) and polysemous terms (`Swift`, `Rust`), while preventing sub-token collisions (`Java` vs `JavaScript`, `Git` vs `GitHub`, `SQL` vs `PL/SQL`).
- **Structured Evidence:** Generates contextual text snippets with character start/end positions for recruiter auditability.
- **Enriched Dataset Output:** Saved to `data/processed/resumes_with_skills.csv` (0.55 MB, 2,484 rows).

---

## 📋 Stage 4: Job Description Analysis
The job analyzer module ([`src/job_analyzer.py`](src/job_analyzer.py)) extracts structured requirements from job descriptions to feed the candidate-matching engine:
- **Unified Skill Taxonomy:** Reuses canonical skill names and aliases from `src.skill_extractor`.
- **Explainable Requirement Separation:** Differentiates essential requirements (*"Requirements:"*, *"must have"*) from optional preferences (*"Preferred:"*, *"nice to have"*) and general context (*"unspecified"*).
- **Recruiter Configuration:** Allows explicit recruiter-confirmed requirements which override automatic heuristics, while reporting unrecognized skills in `unknown_skills`.

---

## ⚙️ Stage 5: Candidate Ranking & Matching Engine

The matching engine ([`src/matching_engine.py`](src/matching_engine.py)) implements an explainable dual-method evaluation pipeline:

### 1. How Candidate Matching Works
The engine evaluates eligible candidates (`is_usable == True` and `is_duplicate == False`, giving 2,481 candidates) using two complementary techniques:

#### Method A: TF-IDF & Cosine Similarity (30% default weight)
- **Concept:** Measures contextual vocabulary overlap between the entire job description and candidate resume.
- **Implementation:** Both the job description and candidate resumes are vectorized in a **shared feature space** using scikit-learn's `TfidfVectorizer`.
- **Preserved Tokens:** A custom token pattern ensures discrete symbols (`C++`, `C#`, `.NET`, `CI/CD`, `PL/SQL`) remain distinct features.
- **Output:** A normalized similarity score scaled between **0.0 and 100.0%**.
- **Important Note:** This measures *textual and topical overlap*, not the candidate's hiring suitability.

#### Method B: Weighted Skill Coverage (70% default weight)
- **Concept:** Directly compares required skills with the candidate's verified skills extracted in Stage 3.
- **Metrics Calculated:**
  - `essential_skills_matched` & `essential_skills_missing`
  - `optional_skills_matched` & `optional_skills_missing`
  - `essential_coverage_percent` = $(Matched_{essential} / Total_{essential}) \times 100$
  - `optional_coverage_percent` = $(Matched_{optional} / Total_{optional}) \times 100$
- **Skill Score Calculation:**
  $$\text{Skill Score} = (\text{Essential Coverage} \times 0.80) + (\text{Optional Coverage} \times 0.20)$$
  *(When no optional skills exist, Essential Coverage contributes 100%).*
- **Documented Fallbacks:**
  - If a job specifies only optional skills: Skill Score = Optional Coverage.
  - If a job description contains no detectable skills: Skill Score falls back to Text Similarity with an explicit warning note.

### 2. Combined Scoring Formula & Default Weights
All scoring parameters are centrally configurable:

$$\text{Combined Score} = (0.70 \times \text{Skill Match Score}) + (0.30 \times \text{Text Similarity Score})$$

```python
DEFAULT_SCORING_CONFIG = {
    "weight_skill": 0.70,           # 70% weight on skill match
    "weight_text": 0.30,            # 30% weight on TF-IDF similarity
    "essential_skill_weight": 0.80, # 80% weight on essential skills (within skill score)
    "optional_skill_weight": 0.20,  # 20% weight on optional skills (within skill score)
}
```

### 3. Deterministic Ranking & Fact-Based Explanations
- **Tie-Breaking:** Candidates are sorted by:
  1. `combined_score` (Descending)
  2. `skill_match_score` (Descending)
  3. `text_similarity_score` (Descending)
  4. `candidate_id` (Ascending)
- **Explainable Feedback:** Every candidate receives a fact-based summary derived directly from matched tokens:
  > *"Matched 4 of 5 essential skills, including Python, SQL. Missing Docker. Optional coverage: 50.0% (including AWS). Text similarity was 62.4%. Review the resume manually to verify proficiency."*

### 4. What the Score Means and Does NOT Mean
- **What It Means:** The score represents an objective, mathematical metric of how closely a candidate's resume vocabulary and reported skills align with the stated job description.
- **What It Does NOT Mean:** The score is **NOT an automated hiring decision**, nor a prediction of job performance. Candidates may possess unlisted skills, or may mention skills without deep proficiency. Rankings are intended as **decision-support tools for human recruiters**.

---

## 🌐 Stage 6: Frontend Web Application & REST API

### Architecture

```
┌──────────────────────┐        HTTP/JSON        ┌──────────────────────┐
│   HTML/CSS/JS        │  ◄──────────────────►    │   FastAPI Backend    │
│   (frontend/)        │                          │   (backend/main.py)  │
│                      │                          │                      │
│  • Job input form    │   GET  /api/health       │  Calls existing:     │
│  • Skills editor     │   GET  /api/categories   │  • job_analyzer      │
│  • Results table     │   POST /api/analyze-job  │  • matching_engine   │
│  • Candidate detail  │   POST /api/rank         │  • skill_extractor   │
│  • Score charts      │   GET  /api/candidate/id │  • text_preprocessor │
└──────────────────────┘                          └──────────────────────┘
```

### API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/health` | System status and dataset statistics |
| `GET` | `/api/categories` | List all job categories with candidate counts |
| `POST` | `/api/analyze-job` | Analyze job description, extract and categorize skills |
| `POST` | `/api/rank` | Rank candidates with filters (category, min score, top N) |
| `GET` | `/api/candidate/{id}` | Retrieve candidate profile details |

### Application Workflow

1. **Open the website** — The dashboard loads and performs a health check.
2. **Paste a job description** — Enter or paste the full job description text.
3. **Analyze** — Click "Analyze Job Description" to extract skills via NLP.
4. **Review & edit skills** — Skills are auto-classified into essential and optional. The recruiter can add, remove, or reclassify skills.
5. **Set filters** — Choose a category filter, minimum score, and result limit.
6. **Analyze & Rank** — Click to rank all eligible candidates using TF-IDF + skill matching.
7. **View results** — See ranked candidates in a sortable table with score distribution chart.
8. **Inspect a candidate** — Click "View" on any candidate to see detailed score breakdown, matched/missing skills, coverage percentages, and explainable summary.

### Design System

The frontend replicates the Frutta-inspired editorial design aesthetic from the reference image:
- **Backgrounds:** Soft strawberry pink (`#FDA2AC`) with ambient radial tints (`#ffd5da`, `#fff0f2`)
- **Primary Headings & Accents:** Vibrant crimson red (`#CB0E31`, `#8f0720`)
- **Botanical Accents & Highlights:** Fresh lime green (`#BAD62C`, `#d2ed42`) with dark olive contrast text (`#232f02`)
- **Typography:** `Bebas Neue` (bold display headlines) + `Plus Jakarta Sans` (interface & body typography)
- **Botanical Citrus Elements:** Crisp, lightweight SVG citrus illustrations (sliced lime wheel, lime wedge, leafy fruit stem, and drifting leaves)
- **Cards & UI:** Frosted milky glassmorphism with subtle crimson borders, smooth pill buttons, and responsive breakpoints at 1024px and 768px

---

## 🗂️ Project Structure
```text
TASK-3/
├── archive/                     # Preserved original Kaggle dataset
│   ├── Resume/
│   │   └── Resume.csv           # Structured dataset (2,484 rows)
│   └── data/
│       └── data/                # 24 category folders containing raw PDFs
│
├── data/
│   ├── raw/                     # Raw data placeholders
│   └── processed/
│       ├── resumes_cleaned.csv  # Stage 2 cleaned & flagged dataset
│       └── resumes_with_skills.csv # Stage 3 enriched dataset with extracted skills
│
├── frontend/                    # Stage 6 — HTML/CSS/JS web interface
│   ├── index.html               # Main HTML page
│   ├── style.css                # Design system & component styles
│   └── script.js                # Frontend logic & API integration
│
├── backend/                     # Stage 6 — FastAPI REST API
│   └── main.py                  # API endpoints connecting to src/ modules
│
├── models/                      # Saved ML models, vectorizers, and metrics
├── notebooks/                   # Jupyter notebooks for EDA and experimentation
│
├── src/                         # Modular source code
│   ├── __init__.py
│   ├── data_loader.py           # Safe dataset loading utility
│   ├── text_preprocessor.py     # Text cleaning, normalization, & deduplication
│   ├── skill_extractor.py       # Rule-based NLP skill extraction & evidence engine
│   ├── job_analyzer.py          # Job description parsing & requirement resolver
│   └── matching_engine.py       # Dual-method matching & ranking engine
│
├── app/                         # (Legacy) Streamlit web application modules
├── scripts/                     # Standalone utility & validation scripts
│   ├── inspect_dataset.py       # Stage 1 dataset inspection script
│   ├── preprocess_dataset.py    # Stage 2 preprocessing execution script
│   ├── extract_skills.py        # Stage 3 skill extraction pipeline script
│   ├── analyze_job_description.py # Stage 4 job description analysis demo
│   └── rank_candidates.py       # Stage 5 candidate ranking demonstration
│
├── tests/                       # Unit tests
│   ├── test_text_preprocessor.py# Tests for preprocessing & deduplication
│   ├── test_skill_extractor.py  # Tests for skill extraction & context guards
│   ├── test_job_analyzer.py     # Tests for JD parsing & recruiter configuration
│   ├── test_matching_engine.py  # Tests for ranking, TF-IDF, coverage, & fallbacks
│   └── test_api.py              # Tests for FastAPI backend endpoints
│
├── requirements.txt             # Project dependencies
├── .gitignore                   # Version control ignore rules
└── README.md                    # Project documentation
```

---

## ⚙️ Environment Setup & Installation

### 1. Prerequisites
- Python 3.10+ (Verified on Python 3.13)
- PowerShell (Windows) or standard terminal

### 2. Create and Activate Virtual Environment
Open PowerShell in the project directory (`TASK-3`):

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🚀 Execution Commands

### 1. Start the Backend API Server
```powershell
python backend/main.py
```
This starts the FastAPI server at `http://127.0.0.1:8000`. The server automatically loads the processed dataset and serves both the API and the frontend static files.

### 2. Access the Frontend
Open your browser and navigate to:
```
http://127.0.0.1:8000
```
The frontend is served as static files by FastAPI. No separate web server is needed.

### 3. Alternative: Start with Uvicorn (Hot Reload)
```powershell
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 4. Run Processing Pipelines (Stages 1–5)
```powershell
python scripts/inspect_dataset.py        # Stage 1: Inspect raw dataset
python scripts/preprocess_dataset.py     # Stage 2: Preprocess resumes
python scripts/extract_skills.py         # Stage 3: Extract skills
python scripts/analyze_job_description.py # Stage 4: Job analysis demo
python scripts/rank_candidates.py        # Stage 5: Ranking demo
```

### 5. Run Full Automated Unit Test Suite
```powershell
python -m unittest discover tests -v
```

---

## ⚠️ Known Limitations

1. **Dataset Scope:** The system is trained on 2,484 resumes across 24 categories from the Kaggle Resume Dataset. It does not accept uploaded resumes in the current version.
2. **Skill Taxonomy:** Skill extraction uses a curated dictionary of ~50 skills. Skills not in the taxonomy will not be automatically matched, though recruiters can manually add them.
3. **Score Interpretation:** Match scores indicate resume-to-job description textual alignment only. They are **not** hiring recommendations, verified competency assessments, or job performance predictions.
4. **No Authentication:** The web application does not implement user authentication or session management.
5. **Single-User Design:** The API maintains a global cached dataset in memory. It is designed for single-user or development use, not production-scale concurrent access.

---

## 🗺️ Project Roadmap
- [x] **Stage 1:** Project Inspection & Dataset Validation
- [x] **Stage 2:** Resume Text Processing & Preprocessing
- [x] **Stage 3:** NLP-Based Skill Extraction
- [x] **Stage 4:** Job Description Analysis
- [x] **Stage 5:** Candidate Matching & Ranking Engine
- [x] **Stage 6:** Frontend Web Application & REST API
- [ ] **Stage 7:** Final Evaluation Report & Project Wrap-Up
