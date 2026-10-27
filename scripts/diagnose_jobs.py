import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, r"c:\Users\SKC\OneDrive\Documents\Desktop\TASK-3")

from src.job_analyzer import analyze_job_description

JD1 = """Job Title: Junior Machine Learning Engineer / AI Engineer Intern

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

JD2 = """Job Title: Data Analyst / Junior Data Scientist

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

JD3 = """Job Title: Python Backend Developer Intern

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

for title, jd in [
    ("Junior Machine Learning Engineer", JD1),
    ("Data Analyst / Junior Data Scientist", JD2),
    ("Python Backend Developer Intern", JD3),
]:
    print("=" * 60)
    print(title)
    res = analyze_job_description(jd)
    print("Essential:", res["essential_skills"])
    print("Optional:", res["optional_skills"])
    print("Unspecified:", res["unspecified_skills"])
    print("Validation messages:", res["validation_messages"])
