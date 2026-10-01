"""
adaptive_interviewer.py
------------------------
The core adaptive component of InterviewIQ.

Instead of pre-generating all interview questions upfront, this module
generates ONE question at a time, using the full interview context so far:
  - resume & JD skills
  - interview type / difficulty / focus areas
  - every previous question, answer, and score
  - which topics the candidate is strong/weak in

Adaptation logic (used both to steer the LLM prompt AND as a deterministic
fallback in Demo Mode / when the LLM is unavailable):
  - If the candidate scored well (>=75) on a topic -> increase difficulty on
    that topic next time it is revisited, or move to a new topic.
  - If the candidate scored poorly (<45) on a topic -> ask an easier
    question on the same topic, or a clarifying follow-up.
  - If the answer was incomplete/vague -> generate a follow-up probing the
    same concept rather than moving on.
"""

import random

from src import llm
from src.sample_data import SAMPLE_HR_QUESTIONS, SAMPLE_QUESTION_BANK

DIFFICULTY_ORDER = ["easy", "medium", "hard"]


def _bump_difficulty(current: str, direction: int) -> str:
    idx = DIFFICULTY_ORDER.index(current) if current in DIFFICULTY_ORDER else 1
    idx = max(0, min(len(DIFFICULTY_ORDER) - 1, idx + direction))
    return DIFFICULTY_ORDER[idx]


def _pick_next_topic(focus_areas, skill_state, asked_topics):
    """Chooses the next topic to probe: prioritize weak topics for
    reinforcement, then untested focus areas, then revisit strong ones."""
    weak = [t for t in skill_state if skill_state[t]["avg_score"] < 45 and skill_state[t]["attempts"] < 2]
    if weak:
        return weak[0]

    untested = [t for t in focus_areas if t not in skill_state]
    if untested:
        return untested[0]

    least_asked = sorted(focus_areas, key=lambda t: skill_state.get(t, {}).get("attempts", 0))
    return least_asked[0] if least_asked else (focus_areas[0] if focus_areas else "Python")


def _deterministic_next_question(context: dict) -> dict:
    """Demo-mode / fallback question generator using the static question
    bank, but still genuinely adaptive based on skill_state."""
    interview_type = context["interview_type"]
    focus_areas = context["focus_areas"] or ["Python", "SQL"]
    skill_state = context["skill_state"]
    asked_questions_text = {q["question"] for q in context["qa_history"]}

    if interview_type == "HR":
        difficulty = context.get("current_difficulty", "medium")
        bank_entry = SAMPLE_HR_QUESTIONS.get(difficulty, SAMPLE_HR_QUESTIONS["medium"])
        return {
            "question": bank_entry["question"],
            "skill": "Behavioral",
            "difficulty": difficulty,
            "question_type": "HR",
            "expected_concepts": bank_entry["expected_concepts"],
            "reason_for_selection": "Standard HR/behavioral question appropriate for this stage of the interview.",
        }

    topic = _pick_next_topic(focus_areas, skill_state, asked_questions_text)
    topic_key = topic if topic in SAMPLE_QUESTION_BANK else random.choice(list(SAMPLE_QUESTION_BANK.keys()))
    state = skill_state.get(topic, {"avg_score": 60, "attempts": 0})

    if state["avg_score"] >= 75:
        difficulty = _bump_difficulty(context.get("current_difficulty", "medium"), +1)
        reason = f"Candidate performed well on {topic} previously; increasing difficulty to probe deeper mastery."
    elif state["avg_score"] < 45 and state["attempts"] > 0:
        difficulty = _bump_difficulty(context.get("current_difficulty", "medium"), -1)
        reason = f"Candidate struggled with {topic}; asking a simpler question on the same topic to gauge foundational understanding."
    else:
        difficulty = context.get("current_difficulty", "medium")
        reason = f"Continuing to assess {topic} at the current difficulty level based on focus areas."

    difficulty = difficulty if difficulty in SAMPLE_QUESTION_BANK.get(topic_key, {}) else "medium"
    bank_entry = SAMPLE_QUESTION_BANK[topic_key].get(difficulty, SAMPLE_QUESTION_BANK[topic_key]["medium"])

    # Avoid immediate repeats
    tries = 0
    while bank_entry["question"] in asked_questions_text and tries < 3:
        alt_difficulty = random.choice(DIFFICULTY_ORDER)
        bank_entry = SAMPLE_QUESTION_BANK[topic_key].get(alt_difficulty, bank_entry)
        difficulty = alt_difficulty
        tries += 1

    q_type = "Technical" if interview_type in ("Technical", "Mixed") else "Behavioral"

    return {
        "question": bank_entry["question"],
        "skill": topic_key,
        "difficulty": difficulty,
        "question_type": q_type,
        "expected_concepts": bank_entry["expected_concepts"],
        "reason_for_selection": reason,
    }


def _build_context_summary(context: dict) -> str:
    lines = []
    for i, qa in enumerate(context["qa_history"][-5:], 1):
        score = qa.get("evaluation", {}).get("overall_score", "N/A")
        lines.append(f"{i}. [{qa.get('skill', '?')} | {qa.get('difficulty', '?')}] Q: {qa['question']}\n"
                      f"   Candidate answer: {qa['answer'][:200]}\n   Score: {score}/100")
    return "\n".join(lines) if lines else "No questions asked yet."


def generate_next_question(context: dict, demo_mode: bool = False) -> dict:
    """
    context must contain:
        interview_type, difficulty (initial), focus_areas (list),
        resume_skills (list), jd_skills (list), qa_history (list of dicts),
        skill_state (dict: topic -> {avg_score, attempts}),
        current_difficulty (str), question_number (int), total_questions (int)

    Returns a dict with keys: question, skill, difficulty, question_type,
    expected_concepts, reason_for_selection.
    """
    if demo_mode or not llm.is_available():
        return _deterministic_next_question(context)

    focus_areas_str = ", ".join(context["focus_areas"]) if context["focus_areas"] else "general based on resume/JD"
    prompt = f"""
You are an expert technical/HR interviewer conducting an adaptive interview.

Interview type: {context['interview_type']}
Overall target difficulty: {context['difficulty']}
Current adaptive difficulty: {context.get('current_difficulty', context['difficulty'])}
Focus areas: {focus_areas_str}
Candidate resume skills: {', '.join(context.get('resume_skills', [])[:20])}
JD required/preferred skills: {', '.join(context.get('jd_skills', [])[:20])}
This is question {context['question_number']} of {context['total_questions']}.

Conversation so far:
{_build_context_summary(context)}

Generate the NEXT interview question. Adapt based on the candidate's
demonstrated performance:
- If they answered well on a topic, ask a harder question (same or new topic).
- If they struggled, ask an easier question or a follow-up on the same concept.
- If an answer was incomplete, consider a clarifying follow-up instead of a
  brand-new topic.
- Do not repeat a question already asked.
- Keep the question realistic for a real interview (concise, one clear question).

Respond with ONLY a JSON object with these exact keys:
- "question": string (the interview question, one question only)
- "skill": string (the primary skill/topic being tested)
- "difficulty": string (one of "easy", "medium", "hard")
- "question_type": string (one of "Technical", "HR", "Behavioral")
- "expected_concepts": list of strings (2-4 key concepts a strong answer should cover)
- "reason_for_selection": string (one sentence explaining why this question was chosen given the candidate's prior performance)
"""
    result, error = llm.generate_json(prompt, temperature=0.8)
    if result is None or not isinstance(result, dict) or "question" not in result:
        fallback = _deterministic_next_question(context)
        fallback["reason_for_selection"] += " (LLM unavailable, used fallback question bank.)"
        return fallback

    result.setdefault("skill", context["focus_areas"][0] if context["focus_areas"] else "General")
    result.setdefault("difficulty", context.get("current_difficulty", "medium"))
    result.setdefault("question_type", "Technical" if context["interview_type"] != "HR" else "HR")
    result.setdefault("expected_concepts", [])
    result.setdefault("reason_for_selection", "Selected based on interview context.")
    return result


def update_skill_state(skill_state: dict, skill: str, score: float) -> dict:
    """Updates the running average score and attempt count for a skill/topic.
    This state drives the adaptive difficulty logic above."""
    if skill not in skill_state:
        skill_state[skill] = {"avg_score": score, "attempts": 1}
    else:
        prev = skill_state[skill]
        new_attempts = prev["attempts"] + 1
        new_avg = ((prev["avg_score"] * prev["attempts"]) + score) / new_attempts
        skill_state[skill] = {"avg_score": round(new_avg, 1), "attempts": new_attempts}
    return skill_state
