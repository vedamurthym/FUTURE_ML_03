"""
Explainable, Rule-Based NLP Skill Extraction Module.

Extracts technical and professional skills from resume text using transparent,
curated regex patterns, word boundaries, and alias normalization.
Includes strict false-positive prevention for single-letter languages (C, R)
and preservation of technical symbols (C++, C#, .NET, Node.js, PL/SQL, CI/CD).
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pandas as pd

# Default paths
DEFAULT_CLEANED_DATA_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "resumes_cleaned.csv"
)
DEFAULT_SKILLS_DATA_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "resumes_with_skills.csv"
)

DISCLAIMER_TEXT = (
    "Text-matching prediction only; not verified candidate ability or qualification."
)


# ============================================================================
# SKILL TAXONOMY & REGEX DICTIONARY
# ============================================================================
# Each entry defines:
# - canonical_name: Standardized display name
# - category: Domain category
# - patterns: List of regex pattern strings with boundaries/context
# ============================================================================

SKILL_DEFINITIONS: List[Dict[str, Any]] = [
    # ------------------------------------------------------------------------
    # 1. Programming Languages
    # ------------------------------------------------------------------------
    {
        "name": "Python",
        "category": "Programming Languages",
        "patterns": [r"\bpython(?:3)?\b"],
    },
    {
        "name": "Java",
        "category": "Programming Languages",
        # Negative lookahead prevents matching JavaScript as Java
        "patterns": [r"\bjava\b(?!\s*script)"],
    },
    {
        "name": "C++",
        "category": "Programming Languages",
        "patterns": [r"\bc\+\+(?!\+)"],
    },
    {
        "name": "C#",
        "category": "Programming Languages",
        # Negative lookahead (?!\w) allows matching before punctuation like comma or period
        "patterns": [r"\bc\#(?!\w)", r"\bc\s*sharp\b"],
    },
    {
        "name": "C",
        "category": "Programming Languages",
        # Strict context matching with negative lookahead (?![+#\w]) prevents false positives
        # from C++, C#, Vitamin C, Grade C, Washington D.C., Option C, etc.
        "patterns": [
            r"\b(?:c\s+programming|c\s+language|ansi\s+c|embedded\s+c|turbo\s+c)\b",
            r"\bc(?![+#\w])\s*/\s*c\+\+",
            r"\b(?:in|with|using)\s+c(?![+#\w])(?:\s+(?:and|or|,)\s+(?:c\+\+|c\#|python|java|sql|assembly))",
            r"\b(?:c\+\+|c\#|python|java|sql|assembly)\s*,\s*c(?![+#\w])",
            r"\bc(?![+#\w])\s*,\s*(?:c\+\+|c\#|python|java|sql|assembly)",
            r"\blanguages?\s*:[^.\n]*?\bc(?![+#\w])",
        ],
    },
    {
        "name": "R",
        "category": "Programming Languages",
        # Strict context matching prevents false positives from R&D, Toys R Us, Model R
        "patterns": [
            r"\b(?:r\s+programming|r\s+language|r\s+packages?|r\s+scripts?|rstudio|the\s+r\s+project|cran)\b",
            r"\b(?:in|with|using)\s+r\b(?:\s+(?:and|or|,)\s+(?:python|sql|sas|matlab|spss|stata))",
            r"\b(?:python|sql|sas|matlab|spss|stata)\s*,\s*r\b",
            r"\br\s*,\s*(?:python|sql|sas|matlab|spss|stata)\b",
        ],
    },
    {
        "name": "JavaScript",
        "category": "Programming Languages",
        # Negative lookbehind (?<!\.) prevents matching .js from Node.js or Vue.js as standalone JS
        "patterns": [r"\bjavascript\b", r"(?<!\.)\bjs\b(?!\s*(?:on|\/))"],
    },
    {
        "name": "TypeScript",
        "category": "Programming Languages",
        "patterns": [r"\btypescript\b", r"\bts\b(?!\s*(?:and|or|\/))"],
    },
    {
        "name": "SQL",
        "category": "Programming Languages",
        # Negative lookbehind prevents matching SQL inside PL/SQL as separate SQL unless standalone
        "patterns": [r"(?<!pl\/)(?<!pl\s\/)\bsql\b(?!\s*server)"],
    },
    {
        "name": "Go",
        "category": "Programming Languages",
        "patterns": [r"\bgolang\b", r"\bgo\s+(?:programming|language)\b"],
    },
    {
        "name": "PHP",
        "category": "Programming Languages",
        "patterns": [r"\bphp\b"],
    },
    {
        "name": "Ruby",
        "category": "Programming Languages",
        "patterns": [r"\bruby\b(?:\s+on\s+rails)?"],
    },
    {
        "name": "Swift",
        "category": "Programming Languages",
        # Context guard prevents false positives on English adjective 'swift' (e.g. swift resolution)
        "patterns": [
            r"\bswift\s+(?:programming|language|developer|development|ios|apps?|code|framework)\b",
            r"\b(?:ios|apple|xcode)\s+swift\b",
            r"\blanguages?\s*:[^.\n]*?\bswift\b",
            r"\bswift\s*(?:,|and|or)\s*(?:objective-c|kotlin|java|python|c\+\+)\b",
            r"\b(?:objective-c|kotlin|java|python|c\+\+)\s*(?:,|and|or)\s*swift\b",
        ],
    },
    {
        "name": "Kotlin",
        "category": "Programming Languages",
        "patterns": [r"\bkotlin\b"],
    },
    {
        "name": "Rust",
        "category": "Programming Languages",
        # Context guard prevents false positives on metal rust / corrosion
        "patterns": [
            r"\brust\s+(?:programming|language|developer|code)\b",
            r"\brust\s*(?:,|and|or)\s*(?:c\+\+|go|python|c)\b",
            r"\b(?:c\+\+|go|python|c)\s*(?:,|and|or)\s*rust\b",
        ],
    },
    {
        "name": "Scala",
        "category": "Programming Languages",
        "patterns": [r"\bscala\b"],
    },

    # ------------------------------------------------------------------------
    # 2. Machine Learning and AI
    # ------------------------------------------------------------------------
    {
        "name": "Machine Learning",
        "category": "Machine Learning & AI",
        "patterns": [r"\bmachine\s+learning\b", r"\bml\s+(?:models?|algorithms?|pipelines?|solutions?)\b"],
    },
    {
        "name": "Deep Learning",
        "category": "Machine Learning & AI",
        "patterns": [r"\bdeep\s+learning\b"],
    },
    {
        "name": "NLP",
        "category": "Machine Learning & AI",
        "patterns": [r"\bnatural\s+language\s+processing\b", r"\bnlp\b"],
    },
    {
        "name": "Generative AI",
        "category": "Machine Learning & AI",
        "patterns": [r"\bgenerative\s+ai\b", r"\bgen\s*ai\b", r"\bllms?\b", r"\blarge\s+language\s+models?\b"],
    },
    {
        "name": "scikit-learn",
        "category": "Machine Learning & AI",
        "patterns": [r"\bscikit\s*-\s*learn\b", r"\bsklearn\b"],
    },
    {
        "name": "TensorFlow",
        "category": "Machine Learning & AI",
        "patterns": [r"\btensorflow\b"],
    },
    {
        "name": "PyTorch",
        "category": "Machine Learning & AI",
        "patterns": [r"\bpytorch\b"],
    },
    {
        "name": "Keras",
        "category": "Machine Learning & AI",
        "patterns": [r"\bkeras\b"],
    },
    {
        "name": "Computer Vision",
        "category": "Machine Learning & AI",
        "patterns": [r"\bcomputer\s+vision\b", r"\bopencv\b"],
    },
    {
        "name": "Data Science",
        "category": "Machine Learning & AI",
        "patterns": [r"\bdata\s+science\b"],
    },

    # ------------------------------------------------------------------------
    # 3. Data Tools
    # ------------------------------------------------------------------------
    {
        "name": "pandas",
        "category": "Data Tools",
        "patterns": [r"\bpandas\b"],
    },
    {
        "name": "NumPy",
        "category": "Data Tools",
        "patterns": [r"\bnumpy\b"],
    },
    {
        "name": "Matplotlib",
        "category": "Data Tools",
        "patterns": [r"\bmatplotlib\b"],
    },
    {
        "name": "Seaborn",
        "category": "Data Tools",
        "patterns": [r"\bseaborn\b"],
    },
    {
        "name": "SciPy",
        "category": "Data Tools",
        "patterns": [r"\bscipy\b"],
    },
    {
        "name": "Power BI",
        "category": "Data Tools",
        "patterns": [r"\bpower\s*bi\b"],
    },
    {
        "name": "Tableau",
        "category": "Data Tools",
        "patterns": [r"\btableau\b"],
    },
    {
        "name": "Excel",
        "category": "Data Tools",
        "patterns": [r"\b(?:microsoft\s+excel|ms\s+excel|excel)\b"],
    },
    {
        "name": "Apache Spark",
        "category": "Data Tools",
        "patterns": [r"\bapache\s+spark\b", r"\bpyspark\b", r"\bspark\s+(?:framework|streaming|core)\b"],
    },
    {
        "name": "Hadoop",
        "category": "Data Tools",
        "patterns": [r"\bhadoop\b"],
    },

    # ------------------------------------------------------------------------
    # 4. Web Development
    # ------------------------------------------------------------------------
    {
        "name": "HTML",
        "category": "Web Development",
        "patterns": [r"\bhtml(?:5)?\b"],
    },
    {
        "name": "CSS",
        "category": "Web Development",
        "patterns": [r"\bcss(?:3)?\b"],
    },
    {
        "name": "React",
        "category": "Web Development",
        "patterns": [r"\breact(?:\.js)?\b(?!\s*native)", r"\breactjs\b"],
    },
    {
        "name": "Node.js",
        "category": "Web Development",
        "patterns": [r"\bnode(?:\.js)?\b", r"\bnodejs\b"],
    },
    {
        "name": "Flask",
        "category": "Web Development",
        "patterns": [r"\bflask\b"],
    },
    {
        "name": "FastAPI",
        "category": "Web Development",
        "patterns": [r"\bfastapi\b"],
    },
    {
        "name": "Django",
        "category": "Web Development",
        "patterns": [r"\bdjango\b"],
    },
    {
        "name": "Angular",
        "category": "Web Development",
        "patterns": [r"\bangular(?:js)?\b"],
    },
    {
        "name": "Vue.js",
        "category": "Web Development",
        "patterns": [r"\bvue(?:\.js)?\b", r"\bvuejs\b"],
    },
    {
        "name": "Spring Boot",
        "category": "Web Development",
        "patterns": [r"\bspring\s*boot\b"],
    },
    {
        "name": "ASP.NET",
        "category": "Web Development",
        "patterns": [r"\basp\.net\b"],
    },
    {
        "name": ".NET",
        "category": "Web Development",
        # Negative lookbehind (?<!\w) allows matching leading .NET and avoids matching inside ASP.NET
        "patterns": [r"(?<!\w)\.net\b", r"\bdotnet\b"],
    },

    # ------------------------------------------------------------------------
    # 5. Databases and Developer Tools
    # ------------------------------------------------------------------------
    {
        "name": "MySQL",
        "category": "Databases & Dev Tools",
        "patterns": [r"\bmysql\b"],
    },
    {
        "name": "PostgreSQL",
        "category": "Databases & Dev Tools",
        "patterns": [r"\bpostgresql\b", r"\bpostgres\b"],
    },
    {
        "name": "MongoDB",
        "category": "Databases & Dev Tools",
        "patterns": [r"\bmongodb\b"],
    },
    {
        "name": "Oracle",
        "category": "Databases & Dev Tools",
        "patterns": [r"\boracle(?:\s+database)?\b"],
    },
    {
        "name": "PL/SQL",
        "category": "Databases & Dev Tools",
        "patterns": [r"\bpl\/sql\b"],
    },
    {
        "name": "Redis",
        "category": "Databases & Dev Tools",
        "patterns": [r"\bredis\b"],
    },
    {
        "name": "Git",
        "category": "Databases & Dev Tools",
        # Negative lookahead prevents matching GitHub/GitLab as Git
        "patterns": [r"\bgit\b(?!\s*(?:hub|lab))"],
    },
    {
        "name": "GitHub",
        "category": "Databases & Dev Tools",
        "patterns": [r"\bgithub\b"],
    },
    {
        "name": "Docker",
        "category": "Databases & Dev Tools",
        "patterns": [r"\bdocker\b"],
    },
    {
        "name": "Kubernetes",
        "category": "Databases & Dev Tools",
        "patterns": [r"\bkubernetes\b", r"\bk8s\b"],
    },
    {
        "name": "Linux",
        "category": "Databases & Dev Tools",
        "patterns": [r"\blinux\b"],
    },
    {
        "name": "AWS",
        "category": "Databases & Dev Tools",
        "patterns": [r"\baws\b", r"\bamazon\s+web\s+services\b"],
    },
    {
        "name": "Azure",
        "category": "Databases & Dev Tools",
        "patterns": [r"\bazure\b", r"\bmicrosoft\s+azure\b"],
    },
    {
        "name": "GCP",
        "category": "Databases & Dev Tools",
        "patterns": [r"\bgcp\b", r"\bgoogle\s+cloud(?:\s+platform)?\b"],
    },
    {
        "name": "CI/CD",
        "category": "Databases & Dev Tools",
        "patterns": [r"\bci\/cd\b"],
    },

    # ------------------------------------------------------------------------
    # 6. Professional Skills
    # ------------------------------------------------------------------------
    {
        "name": "Communication",
        "category": "Professional Skills",
        "patterns": [r"\bcommunication(?:\s+skills)?\b"],
    },
    {
        "name": "Leadership",
        "category": "Professional Skills",
        "patterns": [r"\bleadership\b"],
    },
    {
        "name": "Project Management",
        "category": "Professional Skills",
        "patterns": [r"\bproject\s+management\b"],
    },
    {
        "name": "Problem Solving",
        "category": "Professional Skills",
        "patterns": [r"\bproblem\s*[- ]\s*solving\b"],
    },
    {
        "name": "Teamwork",
        "category": "Professional Skills",
        "patterns": [r"\bteamwork\b"],
    },
    {
        "name": "Critical Thinking",
        "category": "Professional Skills",
        "patterns": [r"\bcritical\s+thinking\b"],
    },
    {
        "name": "Time Management",
        "category": "Professional Skills",
        "patterns": [r"\btime\s+management\b"],
    },
    {
        "name": "Collaboration",
        "category": "Professional Skills",
        "patterns": [r"\bcollaboration\b"],
    },
    {
        "name": "Agile",
        "category": "Professional Skills",
        "patterns": [r"\bagile\b", r"\bscrum\b"],
    },
    {
        "name": "Negotiation",
        "category": "Professional Skills",
        "patterns": [r"\bnegotiation\b"],
    },
]

# Taxonomy mapping category -> list of skill names
SKILL_TAXONOMY: Dict[str, List[str]] = defaultdict(list)
for _s in SKILL_DEFINITIONS:
    SKILL_TAXONOMY[_s["category"]].append(_s["name"])
SKILL_TAXONOMY = dict(SKILL_TAXONOMY)

# Pre-compile regex patterns for maximum efficiency
COMPILED_SKILLS: List[Dict[str, Any]] = []
for skill in SKILL_DEFINITIONS:
    compiled_patterns = [
        re.compile(p, re.IGNORECASE) for p in skill["patterns"]
    ]
    COMPILED_SKILLS.append({
        "name": skill["name"],
        "category": skill["category"],
        "patterns": compiled_patterns,
    })


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

def make_snippet(text: str, start: int, end: int, window: int = 40) -> str:
    """
    Extract a clean contextual snippet surrounding a match.
    Normalizes whitespace and prepends/appends ellipsis when truncated.
    """
    start_pos = max(0, start - window)
    end_pos = min(len(text), end + window)
    raw_snippet = text[start_pos:end_pos].strip()

    # Normalize whitespace inside the snippet
    clean_snippet = re.sub(r"\s+", " ", raw_snippet)

    prefix = "..." if start_pos > 0 else ""
    suffix = "..." if end_pos < len(text) else ""
    return f"{prefix}{clean_snippet}{suffix}"


def extract_skills(text: Optional[str], window_size: int = 40) -> Dict[str, Any]:
    """
    Accept resume text and extract matching technical and professional skills.
    
    Returns structured results:
    - skills: List of unique canonical skill names matched.
    - categories: List of unique categories present.
    - by_category: Dict mapping category name -> list of matched skill names.
    - evidence: Dict mapping canonical skill name -> list of match details:
        [{matched_phrase, snippet, start, end}].
    - total_matches: Total number of mentions found across all skills.
    """
    empty_result = {
        "skills": [],
        "categories": [],
        "by_category": {},
        "evidence": {},
        "total_matches": 0,
    }

    if not text or not isinstance(text, str) or not text.strip():
        return empty_result

    matched_evidence: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    skill_categories: Dict[str, str] = {}
    by_category: Dict[str, List[str]] = defaultdict(list)

    total_matches = 0

    for item in COMPILED_SKILLS:
        skill_name = item["name"]
        category = item["category"]
        skill_categories[skill_name] = category

        # Track spans to prevent duplicate matches from overlapping patterns of the same skill
        seen_spans = set()

        for pattern in item["patterns"]:
            for match in pattern.finditer(text):
                span = match.span()
                # Check for duplicate span
                if span in seen_spans:
                    continue
                seen_spans.add(span)

                start, end = span
                matched_phrase = match.group()
                snippet = make_snippet(text, start, end, window=window_size)

                matched_evidence[skill_name].append({
                    "matched_phrase": matched_phrase,
                    "snippet": snippet,
                    "start": start,
                    "end": end,
                })
                total_matches += 1

        if skill_name in matched_evidence:
            by_category[category].append(skill_name)

    # Sort skills and categories alphabetically for determinism
    sorted_skills = sorted(matched_evidence.keys())
    sorted_categories = sorted(by_category.keys())
    sorted_by_cat = {cat: sorted(by_category[cat]) for cat in sorted_categories}

    return {
        "skills": sorted_skills,
        "categories": sorted_categories,
        "by_category": sorted_by_cat,
        "evidence": dict(matched_evidence),
        "total_matches": total_matches,
    }


def process_resumes_skills(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute skill extraction across all resumes in the dataframe.
    Maintains ID and Category, adds skill counts, skill lists, and JSON-encoded categories.
    Preserves all records for auditability without dropping unselected rows.
    """
    records = []

    for _, row in df.iterrows():
        resume_id = row["ID"]
        category = row["Category"]
        is_usable = bool(row.get("is_usable", True))
        is_duplicate = bool(row.get("is_duplicate", False))

        # Use raw_text if present, fallback to cleaned_text or empty
        text_to_scan = str(row.get("raw_text") or row.get("cleaned_text") or "")

        if is_usable and text_to_scan.strip():
            extracted = extract_skills(text_to_scan)
            skill_list = extracted["skills"]
            skill_count = len(skill_list)
            skills_str = "; ".join(skill_list)
            by_cat_json = json.dumps(extracted["by_category"])
            total_matches = extracted["total_matches"]
        else:
            skill_list = []
            skill_count = 0
            skills_str = ""
            by_cat_json = "{}"
            total_matches = 0

        records.append({
            "ID": resume_id,
            "Category": category,
            "is_usable": is_usable,
            "is_duplicate": is_duplicate,
            "skill_count": skill_count,
            "total_skill_mentions": total_matches,
            "skills": skills_str,
            "skills_by_category": by_cat_json,
            "disclaimer": DISCLAIMER_TEXT,
        })

    return pd.DataFrame(records)


def save_skills_dataset(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Safely save the dataframe with extracted skills to data/processed/.
    """
    dest = Path(output_path).resolve() if output_path else DEFAULT_SKILLS_DATA_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest


def infer_category(
    skills: Optional[Union[List[str], Set[str]]] = None,
    text: Optional[str] = None,
) -> str:
    """
    Infer candidate primary domain category based on detected skills and resume text.
    Uses existing SKILL_DEFINITIONS domain categories.
    Returns 'Other / Unclassified' when no confident domain is identified.
    """
    from collections import Counter

    skill_set = set(skills or [])
    if not skill_set and text and isinstance(text, str) and text.strip():
        extracted = extract_skills(text)
        skill_set = set(extracted.get("skills", []))

    if not skill_set:
        if text and isinstance(text, str):
            lower_text = text.lower()
            if any(kw in lower_text for kw in ["machine learning", "data science", "deep learning", "ai engineer", "artificial intelligence"]):
                return "Machine Learning & AI"
            if any(kw in lower_text for kw in ["web developer", "frontend", "full stack", "backend developer", "web designer"]):
                return "Web Development"
            if any(kw in lower_text for kw in ["data engineer", "data analyst", "business intelligence", "big data"]):
                return "Data Tools"
            if any(kw in lower_text for kw in ["database administrator", "devops", "cloud engineer", "sysadmin"]):
                return "Databases & Dev Tools"
            if any(kw in lower_text for kw in ["software engineer", "programmer", "software developer", "coder"]):
                return "Programming Languages"
        return "Other / Unclassified"

    # Count skills per domain category
    cat_counts: Counter = Counter()
    lookup = {s["name"].lower(): s["category"] for s in SKILL_DEFINITIONS}
    for sk in skill_set:
        cat = lookup.get(str(sk).lower())
        if cat:
            cat_counts[cat] += 1

    if not cat_counts:
        return "Other / Unclassified"

    # Category precedence order for ties (specialized technical domains first)
    PRIORITY = [
        "Machine Learning & AI",
        "Web Development",
        "Data Tools",
        "Databases & Dev Tools",
        "Programming Languages",
        "Professional Skills",
    ]

    # If technical skills are present, prioritize technical domain over generic professional skills
    tech_counts = {k: v for k, v in cat_counts.items() if k != "Professional Skills"}
    target_counts = tech_counts if tech_counts else cat_counts

    if not target_counts:
        return "Other / Unclassified"

    max_count = max(target_counts.values())
    top_cats = [cat for cat, cnt in target_counts.items() if cnt == max_count]

    for p in PRIORITY:
        if p in top_cats:
            return p

    return top_cats[0]

