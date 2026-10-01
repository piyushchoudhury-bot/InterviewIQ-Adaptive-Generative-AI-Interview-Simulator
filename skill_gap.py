"""
skill_gap.py
------------
Compares resume skills vs. JD required/preferred skills vs. skills actually
demonstrated during the interview (via skill_state built up by
adaptive_interviewer.update_skill_state), and classifies each relevant skill
as Strong / Needs Improvement / Missing.

For each weak or missing skill, generates a recommendation (why it matters,
current evidence, what to study, a practice question, and a priority level)
-- using the LLM when available, falling back to templated but still
specific text derived only from data already present in resume/JD/skill_state
(never inventing certifications or requirements not present in the JD).

Also builds a 7-day personalized learning plan from the resulting skill gaps.
"""

from src import llm


def _classify_skills(resume_skills: list, jd_required: list, jd_preferred: list, skill_state: dict) -> dict:
    resume_set = set(s.lower() for s in resume_skills)
    required_set = set(s.lower() for s in jd_required)
    preferred_set = set(s.lower() for s in jd_preferred)
    relevant_skills = sorted(required_set | preferred_set | set(k.lower() for k in skill_state.keys()))

    strong, needs_improvement, missing = [], [], []

    for skill in relevant_skills:
        demonstrated = skill_state.get(skill) or next(
            (v for k, v in skill_state.items() if k.lower() == skill.lower()), None
        )
        on_resume = skill in resume_set
        is_required = skill in required_set

        if demonstrated:
            if demonstrated["avg_score"] >= 70:
                strong.append({"skill": skill, "evidence": f"Average interview score {demonstrated['avg_score']}/100 across {demonstrated['attempts']} question(s)."})
            elif demonstrated["avg_score"] >= 45:
                needs_improvement.append({"skill": skill, "evidence": f"Average interview score {demonstrated['avg_score']}/100 -- partially demonstrated.", "priority": "Medium" if is_required else "Low"})
            else:
                missing.append({"skill": skill, "evidence": f"Attempted in interview but scored only {demonstrated['avg_score']}/100.", "priority": "High" if is_required else "Medium"})
        elif on_resume:
            needs_improvement.append({"skill": skill, "evidence": "Listed on resume but not tested in this interview.", "priority": "Medium" if is_required else "Low"})
        else:
            missing.append({"skill": skill, "evidence": "Not present on resume and not demonstrated in interview.", "priority": "High" if is_required else "Medium"})

    return {"strong": strong, "needs_improvement": needs_improvement, "missing": missing}


def _title_case(skill: str) -> str:
    special = {"sql": "SQL", "aws": "AWS", "gcp": "GCP", "genai": "GenAI", "rag": "RAG", "nlp": "NLP", "etl": "ETL"}
    return special.get(skill.lower(), skill.title())


def _fallback_recommendation(skill: str, evidence: str, jd_data: dict) -> dict:
    topic_map = {
        "python": "Core Python (data structures, OOP, and standard library) -- see resources like the official Python docs or a structured course.",
        "sql": "SQL joins, window functions, and query optimization -- practice on platforms like LeetCode/HackerRank SQL tracks.",
        "machine learning": "Core ML algorithms, evaluation metrics, and the bias-variance tradeoff -- Andrew Ng's Machine Learning course is a strong starting point.",
        "statistics": "Hypothesis testing, p-values, and A/B testing fundamentals.",
        "data structures": "Arrays, linked lists, trees, and hash tables with a focus on time/space complexity.",
        "data engineering": "ETL/ELT pipeline design and data warehousing concepts (star schema, SCD).",
        "cloud": "Core cloud service models (IaaS/PaaS/SaaS) and one provider's fundamentals (AWS/Azure/GCP).",
        "genai": "RAG pipelines, embeddings, and prompt engineering fundamentals.",
    }
    practice_map = {
        "python": "Write a function that removes duplicates from a list while preserving order, without using a set.",
        "sql": "Write a query to find the top 3 highest-paid employees in each department.",
        "machine learning": "Explain how you would handle a dataset with 95% negative class and 5% positive class.",
        "statistics": "Design a simple A/B test to measure the impact of a new button color on click-through rate.",
        "data structures": "Implement a function to check if a binary tree is balanced.",
        "data engineering": "Design a schema to track daily sales across multiple stores and product categories.",
        "cloud": "Compare when you would use a serverless function vs. a containerized service for a given workload.",
        "genai": "Describe the steps you would take to build a RAG-based FAQ chatbot for a company's internal documents.",
    }
    key = skill.lower()
    return {
        "skill": _title_case(skill),
        "why_it_matters": f"'{_title_case(skill)}' appears in the job description's required or preferred skills, or was tested in the interview.",
        "current_evidence": evidence,
        "recommended_topic": topic_map.get(key, f"Review foundational and applied concepts in {_title_case(skill)}."),
        "practice_question": practice_map.get(key, f"Prepare a concise explanation of a real project where you applied {_title_case(skill)}."),
    }


def build_skill_gap_report(resume_data: dict, jd_data: dict, skill_state: dict, demo_mode: bool = False) -> dict:
    resume_skills = resume_data.get("technical_skills", [])
    jd_required = jd_data.get("required_skills", [])
    jd_preferred = jd_data.get("preferred_skills", [])

    classified = _classify_skills(resume_skills, jd_required, jd_preferred, skill_state)

    recommendations = []
    weak_or_missing = classified["needs_improvement"] + classified["missing"]

    for item in weak_or_missing[:8]:
        skill = item["skill"]
        evidence = item["evidence"]
        priority = item.get("priority", "Medium")

        if demo_mode or not llm.is_available():
            rec = _fallback_recommendation(skill, evidence, jd_data)
        else:
            prompt = f"""
A candidate is being evaluated for a role with these job requirements:
Required skills: {', '.join(jd_required)}
Preferred skills: {', '.join(jd_preferred)}

The candidate has a gap in the skill: "{skill}"
Current evidence: {evidence}

Respond with ONLY a JSON object with these exact keys:
- "why_it_matters": one sentence on why this skill matters for this specific role (based only on the JD context given, do not invent requirements)
- "recommended_topic": a specific, concrete sub-topic the candidate should study
- "practice_question": one concrete practice question/exercise for this skill
"""
            result, error = llm.generate_json(prompt, temperature=0.5)
            if result and isinstance(result, dict) and "recommended_topic" in result:
                rec = {
                    "skill": _title_case(skill),
                    "why_it_matters": result.get("why_it_matters", f"'{_title_case(skill)}' is relevant to this role."),
                    "current_evidence": evidence,
                    "recommended_topic": result.get("recommended_topic", ""),
                    "practice_question": result.get("practice_question", ""),
                }
            else:
                rec = _fallback_recommendation(skill, evidence, jd_data)

        rec["priority"] = priority
        recommendations.append(rec)

    return {
        "strong_skills": [{"skill": _title_case(s["skill"]), "evidence": s["evidence"]} for s in classified["strong"]],
        "needs_improvement": [{"skill": _title_case(s["skill"]), "evidence": s["evidence"], "priority": s.get("priority", "Medium")} for s in classified["needs_improvement"]],
        "missing_skills": [{"skill": _title_case(s["skill"]), "evidence": s["evidence"], "priority": s.get("priority", "Medium")} for s in classified["missing"]],
        "recommendations": recommendations,
    }


def build_learning_plan(skill_gap_report: dict, demo_mode: bool = False) -> list:
    """Generates a 7-day personalized learning plan derived from the actual
    skill gaps found above (never invents unrelated topics)."""
    recommendations = skill_gap_report.get("recommendations", [])
    if not recommendations:
        recommendations = [{
            "skill": "General Interview Readiness",
            "recommended_topic": "Review core concepts across your resume's listed skills.",
            "practice_question": "Practice explaining each project on your resume in under 2 minutes.",
        }]

    # Sort so High priority gaps are tackled first
    priority_rank = {"High": 0, "Medium": 1, "Low": 2}
    sorted_recs = sorted(recommendations, key=lambda r: priority_rank.get(r.get("priority", "Medium"), 1))

    plan = []
    for day in range(1, 8):
        rec = sorted_recs[(day - 1) % len(sorted_recs)]
        is_review_day = day == 7
        if is_review_day:
            plan.append({
                "day": day,
                "topic": "Full Review & Mock Interview",
                "tasks": [
                    "Review notes from Days 1-6.",
                    "Re-attempt an InterviewIQ mock interview in the same focus areas.",
                    "Summarize the top 3 improvements made this week.",
                ],
                "practice_question": "Re-answer the hardest question from this week's interview, applying what you learned.",
            })
        else:
            plan.append({
                "day": day,
                "topic": f"{rec['skill']}: {rec.get('recommended_topic', 'Core concepts')}",
                "tasks": [
                    f"Study: {rec.get('recommended_topic', 'core concepts')}",
                    f"Why it matters: {rec.get('why_it_matters', 'Relevant to the target role.')}",
                    "Summarize what you learned in your own words (write 5-10 lines).",
                ],
                "practice_question": rec.get("practice_question", "Practice explaining this topic out loud."),
            })
    return plan
