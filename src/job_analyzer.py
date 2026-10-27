"""
Job Description Analysis Module.

Extracts structured requirements, technical/professional skills, and
qualifications from job descriptions to supply the candidate-matching engine.
Reuses the canonical taxonomy, alias resolution, and boundary rules from
src.skill_extractor and normalization from src.text_preprocessor.
Distinguishes automatic heuristic suggestions from recruiter-confirmed requirements.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from src.text_preprocessor import clean_text
from src.skill_extractor import extract_skills, SKILL_DEFINITIONS


# Regex patterns for section headers
REQ_HEADER_RE = re.compile(
    r"^(?:(?:core|key|basic|job|technical|minimum|role)\s+)?(?:requirements?|qualifications?|must\s+have|required\s+skills?|essential\s+skills?|what\s+you\s+need|what\s+you(\'ll|\s+will)\s+bring)(?:[:\s]|$)",
    re.IGNORECASE,
)
PREF_HEADER_RE = re.compile(
    r"^(?:(?:preferred|desired|optional|bonus|additional)\s+)?(?:qualifications?|skills?|experience|nice\s+to\s+have|good\s+to\s+have|bonus(?:\s+points)?|optional\s+skills?)(?:[:\s]|$)",
    re.IGNORECASE,
)
GEN_HEADER_RE = re.compile(
    r"^(?:about\s+us|about\s+the\s+role|company\s+overview|job\s+summary|responsibilities|duties|what\s+you(\'ll|\s+will)\s+do|benefits|compensation|perks|how\s+to\s+apply)(?:[:\s]|$)",
    re.IGNORECASE,
)

# Inline sentence cues for requirements vs preferences
REQ_INLINE_RE = re.compile(
    r"\b(?:must\s+have|required|essential|mandatory|minimum\s+of|must\s+possess|strict\s+requirement|requires?)\b",
    re.IGNORECASE,
)
PREF_INLINE_RE = re.compile(
    r"\b(?:preferred|nice\s+to\s+have|good\s+to\s+have|plus|bonus|advantage|desired|optional)\b",
    re.IGNORECASE,
)


def validate_job_description(jd_text: Optional[str]) -> Tuple[bool, str]:
    """
    Validate that a job description is not None, empty, or whitespace-only.
    
    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    if jd_text is None:
        return False, "Job description is None. Please provide a valid job description text."

    if not isinstance(jd_text, str) or not jd_text.strip():
        return False, "Job description is empty or whitespace-only. Please provide a detailed job description."

    return True, ""


def resolve_recruiter_skill(skill_input: str) -> Optional[str]:
    """
    Validate and resolve a recruiter-supplied skill against the existing skill taxonomy.
    Reuses canonical names and alias matching from src.skill_extractor.
    
    Returns:
        Canonical skill name if recognized; None if unknown.
    """
    if not skill_input or not isinstance(skill_input, str) or not skill_input.strip():
        return None

    cleaned_input = skill_input.strip()

    # 1. Direct match against canonical taxonomy names (case-insensitive)
    canonical_lookup = {s["name"].lower(): s["name"] for s in SKILL_DEFINITIONS}
    if cleaned_input.lower() in canonical_lookup:
        return canonical_lookup[cleaned_input.lower()]

    # 2. Match via alias patterns from extract_skills
    extracted = extract_skills(cleaned_input)
    matched = extracted.get("skills", [])

    if matched:
        return matched[0]

    return None


def parse_skills_by_section(text: str) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Parse lines/paragraphs and clauses of a job description to identify whether extracted
    skills occur in required sections, preferred sections, or general context.
    
    Returns:
        Tuple of (auto_essential: Set[str], auto_optional: Set[str], auto_unspecified: Set[str])
    """
    auto_essential: Set[str] = set()
    auto_optional: Set[str] = set()
    auto_unspecified: Set[str] = set()

    current_section = "GENERAL"
    raw_lines = text.split("\n")

    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line:
            continue

        # If a line contains multiple inline sections (e.g. "Requirements: Python. Preferred: AWS.")
        # split by boundaries where a new header or bullet starts
        clause_splits = re.split(
            r"(?<=[.;\n])\s+(?=(?:(?:core|key|basic|job|technical|minimum|role)\s+)?(?:requirements?|qualifications?|must\s+have|required\s+skills?|essential\s+skills?)|(?:preferred|desired|optional|bonus|additional|nice\s+to\s+have|good\s+to\s+have)(?:\s+(?:qualifications?|skills?|experience))?[:\s])",
            line,
            flags=re.IGNORECASE,
        )

        for clause in clause_splits:
            stripped = clause.strip()
            if not stripped:
                continue

            clean_header = re.sub(r"^[#*_\-\s]+|[#*_\-\s]+$", "", stripped)

            # Check if this clause sets a new section
            if REQ_HEADER_RE.match(clean_header):
                current_section = "REQUIRED"
                # Strip header prefix to check if skills are in the rest of the clause
                content = re.sub(r"^(?:(?:core|key|basic|job|technical|minimum|role)\s+)?(?:requirements?|qualifications?|must\s+have|required\s+skills?|essential\s+skills?|what\s+you\s+need|what\s+you(\'ll|\s+will)\s+bring)[:\s]*", "", clean_header, flags=re.IGNORECASE).strip()
                if content:
                    extracted_here = extract_skills(content).get("skills", [])
                    for s in extracted_here:
                        auto_essential.add(s)
                continue
            elif PREF_HEADER_RE.match(clean_header):
                current_section = "PREFERRED"
                content = re.sub(r"^(?:(?:preferred|desired|optional|bonus|additional)\s+)?(?:qualifications?|skills?|experience|nice\s+to\s+have|good\s+to\s+have|bonus(?:\s+points)?|optional\s+skills?)[:\s]*", "", clean_header, flags=re.IGNORECASE).strip()
                if content:
                    extracted_here = extract_skills(content).get("skills", [])
                    for s in extracted_here:
                        auto_optional.add(s)
                continue
            elif GEN_HEADER_RE.match(clean_header) or (clean_header.endswith(":") and len(clean_header.split()) <= 4):
                current_section = "GENERAL"
                continue

            # Extract skills present in this clause
            clause_skills = extract_skills(stripped).get("skills", [])
            if not clause_skills:
                continue

            # Check inline phrasing cues
            has_req_inline = bool(REQ_INLINE_RE.search(stripped))
            has_pref_inline = bool(PREF_INLINE_RE.search(stripped))

            for skill in clause_skills:
                if has_req_inline:
                    auto_essential.add(skill)
                elif has_pref_inline:
                    auto_optional.add(skill)
                elif current_section == "REQUIRED":
                    auto_essential.add(skill)
                elif current_section == "PREFERRED":
                    auto_optional.add(skill)
                else:
                    auto_unspecified.add(skill)

    # Resolution: essential takes precedence over optional and unspecified
    auto_optional -= auto_essential
    auto_unspecified -= (auto_essential | auto_optional)

    return auto_essential, auto_optional, auto_unspecified



def validate_recruiter_configuration(
    config: Dict[str, Any]
) -> Tuple[List[str], List[str], List[Dict[str, str]]]:
    """
    Validate recruiter-provided configuration of essential and optional skills.
    
    Returns:
        Tuple of:
        - confirmed_essential: List of recognized canonical skill names
        - confirmed_optional: List of recognized canonical skill names
        - unknown_skills: List of dicts reporting unrecognized skill entries
    """
    confirmed_essential: List[str] = []
    confirmed_optional: List[str] = []
    unknown_skills: List[Dict[str, str]] = []

    essential_inputs = config.get("essential_skills", [])
    optional_inputs = config.get("optional_skills", [])

    # Process essential entries
    for item in essential_inputs:
        resolved = resolve_recruiter_skill(str(item))
        if resolved:
            if resolved not in confirmed_essential:
                confirmed_essential.append(resolved)
        else:
            unknown_skills.append({
                "raw_skill": str(item),
                "specified_as": "essential",
                "message": f"Skill '{item}' is not recognized in the current skill taxonomy.",
            })

    # Process optional entries
    for item in optional_inputs:
        resolved = resolve_recruiter_skill(str(item))
        if resolved:
            if resolved not in confirmed_optional and resolved not in confirmed_essential:
                confirmed_optional.append(resolved)
        else:
            unknown_skills.append({
                "raw_skill": str(item),
                "specified_as": "optional",
                "message": f"Skill '{item}' is not recognized in the current skill taxonomy.",
            })

    return sorted(confirmed_essential), sorted(confirmed_optional), unknown_skills


def analyze_job_description(
    jd_text: Optional[str],
    recruiter_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Analyze a job description text, extract skills, and resolve requirements.
    
    Args:
        jd_text: Raw job description text string.
        recruiter_config: Optional dict with 'essential_skills' and 'optional_skills'.
        
    Returns:
        Structured dict with extraction results, validation messages, and categorized skills.
    """
    # 1. Validate JD text
    is_valid, error_msg = validate_job_description(jd_text)
    if not is_valid:
        return {
            "is_valid": False,
            "original_jd": jd_text or "",
            "cleaned_jd": "",
            "extracted_skills": [],
            "by_category": {},
            "evidence": {},
            "essential_skills": [],
            "optional_skills": [],
            "unspecified_skills": [],
            "auto_suggested_essential": [],
            "auto_suggested_optional": [],
            "confirmed_essential": [],
            "confirmed_optional": [],
            "unknown_skills": [],
            "has_skills": False,
            "validation_messages": [error_msg],
        }

    original_jd = jd_text.strip()
    cleaned_jd = clean_text(original_jd)

    # 2. Extract all skills from the entire JD using the existing skill extractor
    extraction_results = extract_skills(original_jd)
    all_extracted_skills = extraction_results.get("skills", [])
    by_category = extraction_results.get("by_category", {})
    evidence = extraction_results.get("evidence", {})

    # 3. Parse automatic requirement cues (Required vs. Preferred vs. General)
    auto_essential, auto_optional, auto_unspecified = parse_skills_by_section(original_jd)

    # 4. Handle recruiter-provided configuration if present
    confirmed_essential: List[str] = []
    confirmed_optional: List[str] = []
    unknown_skills: List[Dict[str, str]] = []

    if recruiter_config and isinstance(recruiter_config, dict):
        confirmed_essential, confirmed_optional, unknown_skills = (
            validate_recruiter_configuration(recruiter_config)
        )

    # 5. Determine final essential and optional skills
    # If recruiter provided confirmed requirements, they take precedence
    validation_messages: List[str] = []

    if confirmed_essential or confirmed_optional:
        essential_skills = sorted(list(confirmed_essential))
        optional_skills = sorted(list(confirmed_optional))
        validation_messages.append("Using recruiter-confirmed requirements for candidate ranking.")
    else:
        # Fallback to automatic heuristic suggestions
        essential_skills = sorted(list(auto_essential))
        optional_skills = sorted(list(auto_optional))
        if essential_skills or optional_skills:
            validation_messages.append(
                "Using automatically suggested requirements derived from explicit job description phrases."
            )
        else:
            validation_messages.append(
                "Warning: No explicit skill requirements could be identified in the job description. "
                "Please check the job description (e.g. add a 'Requirements:' section) or configure essential skills manually."
            )

    has_skills = bool(all_extracted_skills or essential_skills)

    # 6. Validate if no detectable skills were found
    if not has_skills:
        validation_messages.append(
            "No technical or professional skills were detected in the provided job description. "
            "Please provide a more detailed description including specific technologies, tools, "
            "and qualifications, or specify essential skills manually using recruiter configuration."
        )

    if unknown_skills:
        unknown_names = [u["raw_skill"] for u in unknown_skills]
        validation_messages.append(
            f"The following recruiter-supplied skills were not recognized in the skill taxonomy: {', '.join(unknown_names)}. "
            "They will be recorded for review but cannot be matched automatically."
        )

    warnings = [m for m in validation_messages if "Warning" in m or "No technical" in m or "No explicit" in m]

    return {
        "is_valid": True,
        "original_jd": original_jd,
        "cleaned_jd": cleaned_jd,
        "extracted_skills": all_extracted_skills,
        "by_category": by_category,
        "evidence": evidence,
        "essential_skills": essential_skills,
        "optional_skills": optional_skills,
        "unspecified_skills": sorted(list(auto_unspecified)),
        "auto_suggested_essential": sorted(list(auto_essential)),
        "auto_suggested_optional": sorted(list(auto_optional)),
        "confirmed_essential": confirmed_essential,
        "confirmed_optional": confirmed_optional,
        "unknown_skills": unknown_skills,
        "has_skills": has_skills,
        "warnings": warnings,
        "validation_messages": validation_messages,
    }
