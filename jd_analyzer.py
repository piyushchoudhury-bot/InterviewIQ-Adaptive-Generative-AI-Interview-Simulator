"""
jd_analyzer.py
--------------
Extracts structured information from a Job Description: required skills,
preferred skills, responsibilities, qualifications, tools, and experience
requirements. Mirrors resume_analyzer.py's LLM-first / keyword-fallback
design so the app never crashes if the LLM is unavailable.
"""

import re

from src import llm
from src.resume_analyzer import KNOWN_SKILLS, _looks_like_skill_token
from src.sample_data import SAMPLE_JD_DATA


def _extract_known_skills(text: str):
    text_lower = text.lower()
    found = []
    for skill in KNOWN_SKILLS:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill)
    return found


JD_SECTION_HEADERS = {
    "required_skills": r"^(required skills?|must[- ]have skills?|requirements?)\s*:?\s*$",
    "preferred_skills": r"^(preferred skills?|nice[- ]to[- ]have skills?|bonus( points)?)\s*:?\s*$",
    "responsibilities": r"^(responsibilities|key responsibilities|role|what you.?ll do|duties)\s*:?\s*$",
    "qualifications": r"^(qualifications?|education|eligibility)\s*:?\s*$",
    "tools": r"^(tools|tools\s*/\s*technolog(y|ies)|technolog(y|ies))\s*:?\s*$",
}


def _split_into_sections(text: str) -> dict:
    """Splits JD text into sections by detecting common header lines, so we
    can pull every line under 'Required Skills:', 'Responsibilities:', etc.
    instead of relying on a single regex spanning the whole document."""
    lines = [l.rstrip() for l in text.split("\n")]
    sections = {k: [] for k in JD_SECTION_HEADERS}
    current = None
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        matched_header = None
        if len(line) <= 60:
            for key, pattern in JD_SECTION_HEADERS.items():
                if re.match(pattern, line, re.I):
                    matched_header = key
                    break
        if matched_header:
            current = matched_header
            continue
        if current:
            sections[current].append(line.strip("-*•\t "))
    return sections


def _split_list_like_line(line: str):
    parts = re.split(r"[,|/•·]|\s{2,}", line)
    cleaned = [p.strip(" .;:-") for p in parts]
    return [p for p in cleaned if _looks_like_skill_token(p)]


def _fallback_extraction(jd_text: str) -> dict:
    lines = [l.strip("-*• \t") for l in jd_text.split("\n") if l.strip()]
    sections = _split_into_sections(jd_text)

    all_skills = _extract_known_skills(jd_text)

    # Prefer explicit "Required/Preferred Skills" sections; fall back to a
    # broad regex span; fall back further to "all skills found anywhere".
    if sections["required_skills"]:
        required_text = " ".join(sections["required_skills"])
        required_skills = _extract_known_skills(required_text)
        listed_required = []
        for line in sections["required_skills"]:
            listed_required.extend(_split_list_like_line(line))
        required_skills = list(dict.fromkeys(required_skills + [s for s in listed_required if len(s) < 40]))
    else:
        required_section = re.search(r"required skills?:?(.*?)(preferred|qualification|responsibilit|$)", jd_text, re.I | re.S)
        required_skills = _extract_known_skills(required_section.group(1)) if required_section else all_skills

    if sections["preferred_skills"]:
        preferred_text = " ".join(sections["preferred_skills"])
        preferred_skills = _extract_known_skills(preferred_text)
        listed_preferred = []
        for line in sections["preferred_skills"]:
            listed_preferred.extend(_split_list_like_line(line))
        preferred_skills = list(dict.fromkeys(preferred_skills + [s for s in listed_preferred if len(s) < 40]))
    else:
        preferred_section = re.search(r"preferred skills?:?(.*?)(qualification|responsibilit|tools|$)", jd_text, re.I | re.S)
        preferred_skills = _extract_known_skills(preferred_section.group(1)) if preferred_section else []

    preferred_skills = [s for s in preferred_skills if s.lower() not in {r.lower() for r in required_skills}]
    if not required_skills:
        required_skills = all_skills

    responsibilities = sections["responsibilities"] or [
        l for l in lines if re.search(r"^(build|develop|design|write|collaborat|maintain|analyz|create|manage|own|support|lead|work with|partner)", l, re.I)
    ][:10]
    qualifications = sections["qualifications"] or [
        l for l in lines if re.search(r"\b(degree|bachelor|master|years? of experience|qualification)\b", l, re.I)
    ]
    tools = sections["tools"] or all_skills

    return {
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "responsibilities": responsibilities if responsibilities else ["Not clearly specified"],
        "qualifications": qualifications if qualifications else ["Not clearly specified"],
        "tools": tools,
        "experience_requirement": next((l for l in lines if re.search(r"\d+\+?\s*years?", l, re.I)), "Not specified"),
    }


def analyze_jd(jd_text: str, demo_mode: bool = False) -> dict:
    """Returns a structured dict describing the job description. Never raises."""
    if demo_mode:
        return dict(SAMPLE_JD_DATA)

    if not jd_text or not jd_text.strip():
        return _fallback_extraction("")

    prompt = f"""
Extract structured information from the following job description. Respond
with ONLY a JSON object with these exact keys:
- "required_skills": list of strings (must-have skills/technologies)
- "preferred_skills": list of strings (nice-to-have skills, distinct from required_skills)
- "responsibilities": list of strings
- "qualifications": list of strings (education, years of experience, etc.)
- "tools": list of strings (specific tools/technologies mentioned)
- "experience_requirement": string (e.g. "2-4 years" or "Not specified")

Only include information that is explicitly present in the text below. Do not
invent requirements, certifications, or tools that are not mentioned.

JOB DESCRIPTION:
\"\"\"{jd_text[:8000]}\"\"\"
"""
    result, error = llm.generate_json(prompt, temperature=0.2)
    if result is None or not isinstance(result, dict):
        fallback = _fallback_extraction(jd_text)
        fallback["_llm_error"] = error
        return fallback

    for key in ["required_skills", "preferred_skills", "responsibilities", "qualifications", "tools"]:
        if key not in result:
            result[key] = []
    if "experience_requirement" not in result:
        result["experience_requirement"] = "Not specified"
    return result


def compute_match(resume_data: dict, jd_data: dict, resume_text: str, jd_text: str) -> dict:
    """
    Computes the Resume-JD match score:
        Overall Match = 60% Semantic Similarity + 40% Skill Coverage
    This is clearly an application-generated heuristic, NOT an official ATS score.
    """
    from src import embeddings

    resume_skills = set(s.strip().lower() for s in resume_data.get("technical_skills", []))
    required_skills = set(s.strip().lower() for s in jd_data.get("required_skills", []))
    preferred_skills = set(s.strip().lower() for s in jd_data.get("preferred_skills", []))

    matching_skills = sorted(resume_skills & required_skills, key=str)
    missing_required = sorted(required_skills - resume_skills, key=str)
    missing_preferred = sorted(preferred_skills - resume_skills, key=str)

    skill_coverage = (len(matching_skills) / len(required_skills) * 100) if required_skills else 0.0

    semantic_score = embeddings.cosine_similarity_score(resume_text, jd_text)
    used_fallback_similarity = False
    if semantic_score is None:
        semantic_score = embeddings.keyword_overlap_score(resume_text, jd_text)
        used_fallback_similarity = True

    overall_match = round(0.6 * semantic_score + 0.4 * skill_coverage, 1)

    def title_case_list(skills):
        # Recover nicer casing from KNOWN_SKILLS where possible
        lookup = {s.lower(): s for s in KNOWN_SKILLS}
        return [lookup.get(s, s.title()) for s in skills]

    experience_alignment = "Not enough information to assess" 
    exp_req = jd_data.get("experience_requirement", "")
    if exp_req and exp_req.lower() not in ("not specified", ""):
        experience_alignment = f"JD requires: {exp_req}. Review candidate's experience section manually for exact alignment."

    return {
        "overall_match": overall_match,
        "semantic_similarity": semantic_score,
        "used_fallback_similarity": used_fallback_similarity,
        "skill_coverage": round(skill_coverage, 1),
        "matching_skills": title_case_list(matching_skills),
        "missing_required_skills": title_case_list(missing_required),
        "missing_preferred_skills": title_case_list(missing_preferred),
        "experience_alignment": experience_alignment,
    }
