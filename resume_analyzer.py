"""
resume_analyzer.py
-------------------
Extracts structured information from raw resume text: name, education,
experience, projects, technical skills, soft skills, certifications, tools.

Primary path: ask Gemini to extract structured JSON.
Fallback path: keyword/regex-based extraction against a curated skills list,
used in Demo Mode or whenever the LLM call fails, so the app always produces
a usable result.
"""

import re

from src import llm
from src.sample_data import SAMPLE_RESUME_DATA

KNOWN_SKILLS = [
    "Python", "Java", "C++", "C", "JavaScript", "TypeScript", "R", "SQL", "NoSQL",
    "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", "Keras",
    "Machine Learning", "Deep Learning", "Statistics", "Data Structures",
    "Algorithms", "Data Engineering", "ETL", "Spark", "Hadoop", "Kafka",
    "Airflow", "Power BI", "Tableau", "Plotly", "Matplotlib", "Excel",
    "AWS", "Azure", "GCP", "Cloud", "Docker", "Kubernetes",
    "GenAI", "LLM", "RAG", "LangChain", "Prompt Engineering", "NLP",
    "Computer Vision", "Git", "GitHub", "MongoDB", "MySQL", "PostgreSQL",
    "REST API", "Flask", "Django", "FastAPI", "Streamlit", "HTML", "CSS",
    "React", "Node.js", "Linux", "Agile", "Scrum",
]

SOFT_SKILL_KEYWORDS = [
    "communication", "teamwork", "leadership", "problem-solving", "problem solving",
    "adaptability", "time management", "collaboration", "critical thinking",
    "creativity", "attention to detail", "work ethic", "presentation",
]


def _extract_known_skills(text: str):
    text_lower = text.lower()
    found = []
    for skill in KNOWN_SKILLS:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill)
    return found


def _extract_soft_skills(text: str):
    text_lower = text.lower()
    return [s.title() for s in SOFT_SKILL_KEYWORDS if s in text_lower]


SECTION_HEADERS = {
    "education": r"^(education|academic background|academic qualifications?)\s*:?\s*$",
    "experience": r"^(experience|work experience|professional experience|employment history|internships?)\s*:?\s*$",
    "projects": r"^(projects?|academic projects?|personal projects?|key projects?)\s*:?\s*$",
    "technical_skills": r"^(technical skills|skills|core competencies|key skills|technologies)\s*:?\s*$",
    "soft_skills": r"^(soft skills|interpersonal skills)\s*:?\s*$",
    "certifications": r"^(certifications?|licenses?( and certifications?)?|courses?)\s*:?\s*$",
    "tools": r"^(tools|tools\s*/\s*technologies|tools and technologies)\s*:?\s*$",
}


def _split_into_sections(text: str) -> dict:
    """Splits resume text into sections by detecting common section-header
    lines (EDUCATION, EXPERIENCE, PROJECTS, SKILLS, etc.), which is far more
    reliable than single-line keyword matching for real-world resumes."""
    lines = [l.rstrip() for l in text.split("\n")]
    sections = {k: [] for k in SECTION_HEADERS}
    current = None
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        matched_header = None
        # A header line is short and matches one of the known header patterns
        if len(line) <= 60:
            for key, pattern in SECTION_HEADERS.items():
                if re.match(pattern, line, re.I):
                    matched_header = key
                    break
        if matched_header:
            current = matched_header
            continue
        if current:
            sections[current].append(line.strip("-*•\t "))
    return sections


SKILL_TOKEN_STOPWORDS = {
    "experience", "years", "with", "and", "or", "including", "such", "as", "ability",
    "familiarity", "knowledge", "proficiency", "strong", "working", "environment",
    "understanding", "skills", "skill", "etc", "plus", "preferred", "required",
    "solid", "good", "excellent", "hands-on", "exposure", "basic", "advanced",
    "of", "in", "for", "the", "a", "an", "using",
}


def _looks_like_skill_token(token: str) -> bool:
    """Filters raw split fragments down to plausible technology/skill names,
    rejecting sentence fragments like 'Experience with Pandas' or unbalanced
    parenthetical leftovers like '(ETL'."""
    token = token.strip()
    if not (1 < len(token) <= 30):
        return False
    if token.count("(") != token.count(")"):
        return False
    words = token.lower().split()
    if len(words) > 3:
        return False
    if any(w in SKILL_TOKEN_STOPWORDS for w in words):
        return False
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9+.#/&\- ]*$", token):
        return False
    return True


def _split_list_like_line(line: str):
    """Splits a comma/pipe/bullet-separated line of skills into individual
    items, then filters out anything that doesn't look like a real skill
    name (e.g. leftover sentence fragments)."""
    parts = re.split(r"[,|/•·]|\s{2,}", line)
    cleaned = [p.strip(" .;:-") for p in parts]
    return [p for p in cleaned if _looks_like_skill_token(p)]


def _fallback_extraction(resume_text: str) -> dict:
    """Deterministic regex/section-based extraction used when the LLM is
    unavailable. First tries to detect real resume section headers
    (EDUCATION, EXPERIENCE, PROJECTS, SKILLS, CERTIFICATIONS...) and pulls
    every line under them; falls back to loose keyword matching only for
    whatever a section-based pass could not find."""
    lines = [l.strip() for l in resume_text.split("\n") if l.strip()]
    name = lines[0][:60] if lines else "Candidate"

    sections = _split_into_sections(resume_text)

    education = sections["education"] or [
        l for l in lines if re.search(r"\b(b\.?tech|m\.?tech|bachelor|master|university|institute|college|cgpa|degree)\b", l, re.I)
    ]
    experience = sections["experience"] or [
        l for l in lines if re.search(r"\b(intern|experience|worked|engineer|analyst|developer)\b", l, re.I)
    ]
    projects = sections["projects"] or [l for l in lines if re.search(r"\bproject\b", l, re.I)]
    certifications = sections["certifications"] or [
        l for l in lines if re.search(r"\b(certificat|certified|course completion)\b", l, re.I)
    ]

    # Technical skills: union of (a) known-skill keyword matches anywhere in
    # the resume, and (b) whatever is literally listed under a "Skills"
    # section header, even if it's not in our curated KNOWN_SKILLS list.
    keyword_skills = _extract_known_skills(resume_text)
    listed_skills = []
    for line in sections["technical_skills"]:
        listed_skills.extend(_split_list_like_line(line))
    # De-duplicate case-insensitively while preserving first-seen casing
    seen = set()
    technical_skills = []
    for s in keyword_skills + listed_skills:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            technical_skills.append(s)

    tools = sections["tools"] or [s for s in technical_skills if s in
                                   ["Git", "GitHub", "Docker", "Kubernetes", "Excel", "Power BI", "Tableau"]]
    tools_flat = []
    for t in tools:
        tools_flat.extend(_split_list_like_line(t) if len(t) > 40 or "," in t else [t])

    if sections["soft_skills"]:
        soft_skills = []
        for line in sections["soft_skills"]:
            soft_skills.extend(_split_list_like_line(line))
        soft_skills = [s.title() for s in soft_skills]
    else:
        soft_skills = _extract_soft_skills(resume_text)

    return {
        "name": name,
        "education": education[:6] if education else ["Not clearly specified"],
        "experience": experience[:8] if experience else ["Not clearly specified"],
        "projects": projects[:8] if projects else ["Not clearly specified"],
        "technical_skills": technical_skills,
        "soft_skills": soft_skills,
        "certifications": certifications if certifications else [],
        "tools": tools_flat if tools_flat else [s for s in technical_skills if s in
                 ["Git", "GitHub", "Docker", "Kubernetes", "Excel", "Power BI", "Tableau"]],
    }


def analyze_resume(resume_text: str, demo_mode: bool = False) -> dict:
    """Returns a structured dict describing the resume. Never raises."""
    if demo_mode:
        return dict(SAMPLE_RESUME_DATA)

    if not resume_text or not resume_text.strip():
        return _fallback_extraction("")

    prompt = f"""
Extract structured information from the following resume text. Respond with
ONLY a JSON object with these exact keys:
- "name": string
- "education": list of strings (degree, institution, year, grade if present)
- "experience": list of strings (role, company, duration, brief description)
- "projects": list of strings (project name + one-line description)
- "technical_skills": list of strings (programming languages, frameworks, technologies)
- "soft_skills": list of strings
- "certifications": list of strings
- "tools": list of strings (software/tools used)

If a section is not present in the resume, return an empty list for it (not null).
Do not invent information that is not present in the resume text.

RESUME TEXT:
\"\"\"{resume_text[:8000]}\"\"\"
"""
    result, error = llm.generate_json(prompt, temperature=0.2)
    if result is None or not isinstance(result, dict):
        fallback = _fallback_extraction(resume_text)
        fallback["_llm_error"] = error
        return fallback

    for key in ["name", "education", "experience", "projects", "technical_skills", "soft_skills", "certifications", "tools"]:
        if key not in result:
            result[key] = [] if key != "name" else "Candidate"
    return result
