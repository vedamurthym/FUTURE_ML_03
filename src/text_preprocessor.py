"""
Resume Text Preprocessing Module.
Handles text normalization, cleaning, technical term preservation,
HTML fallback extraction, usability flagging, and safe deduplication.
"""

import html
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

# Default destination for cleaned data
DEFAULT_PROCESSED_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "resumes_cleaned.csv"
)

# Common technical terms with symbols that must be preserved during NLP preprocessing
TECHNICAL_TERM_PATTERNS: List[Tuple[str, str, str]] = [
    # (regex_pattern, placeholder, restored_value)
    (r"\basp\.net\b", "techterm_aspdotnet", "asp.net"),
    (r"(?<!\w)\.net\b", "techterm_dotnet", ".net"),
    (r"\bc\+\+", "techterm_cpp", "c++"),
    (r"\bc\#", "techterm_csharp", "c#"),
    (r"\bnode\.js\b", "techterm_nodejs", "node.js"),
    (r"\breact\.js\b", "techterm_reactjs", "react.js"),
    (r"\bvue\.js\b", "techterm_vuejs", "vue.js"),
    (r"\bci\/cd\b", "techterm_cicd", "ci/cd"),
    (r"\btcp\/ip\b", "techterm_tcpip", "tcp/ip"),
    (r"\bpl\/sql\b", "techterm_plsql", "pl/sql"),
    (r"\br\&d\b", "techterm_rnd", "r&d"),
]


def extract_text_from_html(html_content: Optional[str]) -> str:
    """
    Safely strip HTML markup, style, and script tags, decode entities,
    and return normalized plain text.
    """
    if not html_content or not isinstance(html_content, str):
        return ""

    # Remove script and style elements
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html_content, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)

    # Replace all HTML tags with whitespace
    text = re.sub(r"<[^>]+>", " ", text)

    # Unescape HTML entities (e.g., &amp; -> &, &nbsp; -> ' ')
    text = html.unescape(text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text(text: Optional[str], preserve_tech_terms: bool = True) -> str:
    """
    Clean and normalize resume text for downstream NLP tasks.
    
    Processing steps:
    1. Null/empty check -> returns empty string.
    2. Lowercase normalization.
    3. Remove any residual HTML markup.
    4. Protect technical terms (e.g. C++, C#, .NET, Node.js) with placeholders.
    5. Remove URLs and email addresses.
    6. Replace bullet points, formatting artifacts, and general punctuation with spaces.
    7. Restore protected technical terms.
    8. Normalize multiple whitespace characters into a single space and strip edges.
    
    Args:
        text: Input string or text candidate.
        preserve_tech_terms: Whether to protect technical terms containing symbols.
        
    Returns:
        Cleaned, normalized string.
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    if not text.strip():
        return ""

    # 1. Convert to lowercase
    cleaned = text.lower()

    # 2. Strip any embedded HTML tags if present
    if "<" in cleaned and ">" in cleaned:
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        cleaned = html.unescape(cleaned)

    # 3. Protect technical terms with placeholders
    restore_mapping: Dict[str, str] = {}
    if preserve_tech_terms:
        for pattern, placeholder, restored_val in TECHNICAL_TERM_PATTERNS:
            if re.search(pattern, cleaned):
                cleaned = re.sub(pattern, placeholder, cleaned)
                restore_mapping[placeholder] = restored_val

    # 4. Remove URLs and email addresses
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)
    cleaned = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", cleaned)

    # 5. Remove bullet points, unwanted symbols, and punctuation
    # Allow alphanumeric characters, underscores, and spaces
    # Protected technical placeholders contain only word characters
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)

    # 6. Restore technical terms
    if preserve_tech_terms and restore_mapping:
        for placeholder, original_term in restore_mapping.items():
            cleaned = re.sub(r"\b" + placeholder + r"\b", original_term, cleaned)

    # 7. Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def resolve_resume_text(resume_str: Optional[str], resume_html: Optional[str]) -> Tuple[str, str, bool]:
    """
    Resolve the best available text representation for a resume record.
    
    Checks Resume_str first. If Resume_str is empty or whitespace-only,
    inspects Resume_html for extractable content.
    
    Returns:
        Tuple of (raw_text, cleaned_text, is_usable)
    """
    raw_text = ""
    is_usable = False

    # Check primary plain-text string
    if resume_str and isinstance(resume_str, str) and resume_str.strip():
        raw_text = resume_str.strip()
    elif resume_html and isinstance(resume_html, str) and resume_html.strip():
        # Fallback inspection of HTML
        extracted = extract_text_from_html(resume_html)
        if extracted.strip():
            raw_text = extracted.strip()

    cleaned_text = clean_text(raw_text)

    # Usable if cleaned text has meaningful content
    if cleaned_text:
        is_usable = True

    return raw_text, cleaned_text, is_usable


def identify_duplicates(df: pd.DataFrame, text_column: str = "cleaned_text") -> pd.Series:
    """
    Identify duplicate resume texts among usable records.
    The first occurrence is considered primary (False), while subsequent
    occurrences of identical text are flagged as duplicates (True).
    Unusable records are marked False for duplicate flag.
    """
    is_dup = pd.Series(False, index=df.index)
    usable_mask = df["is_usable"] if "is_usable" in df.columns else pd.Series(True, index=df.index)

    # Only evaluate duplicates among usable records
    usable_subset = df[usable_mask]
    dup_mask = usable_subset.duplicated(subset=[text_column], keep="first")
    is_dup.loc[usable_subset.index] = dup_mask

    return is_dup


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute full Stage 2 preprocessing pipeline across the resume dataframe.
    
    Preserves original columns ('ID', 'Category'), resolves raw resume text
    with HTML fallback inspection, produces cleaned text for NLP, and flags
    both usability and duplicate status.
    
    Returns:
        pd.DataFrame with columns:
        ['ID', 'Category', 'raw_text', 'cleaned_text', 'is_usable', 'is_duplicate']
    """
    records = []

    for idx, row in df.iterrows():
        resume_id = row["ID"] if "ID" in row else idx
        category = row["Category"] if "Category" in row else "UNKNOWN"
        r_str = row.get("Resume_str", None)
        r_html = row.get("Resume_html", None)

        raw_text, cleaned_text, is_usable = resolve_resume_text(r_str, r_html)

        records.append({
            "ID": resume_id,
            "Category": category,
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "is_usable": is_usable,
        })

    processed_df = pd.DataFrame(records)

    # Flag duplicates among usable records
    processed_df["is_duplicate"] = identify_duplicates(processed_df, text_column="cleaned_text")

    return processed_df


def save_processed_data(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Safely save the processed dataframe to data/processed/.
    Ensures parent directories exist and does not touch raw files.
    """
    dest = Path(output_path).resolve() if output_path else DEFAULT_PROCESSED_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest


def get_modeling_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience helper to extract the subset of records suitable for model training.
    Filters out unusable (empty) and duplicate resumes.
    """
    return df[df["is_usable"] & (~df["is_duplicate"])].copy()
