"""
analytics.py
------------
Deterministic, purely computed communication metrics for a candidate's text
answer. These are NOT LLM inferences -- they are measured directly from the
text, and are clearly labeled as such in the UI so users understand the
difference between measured signals and LLM judgment (see evaluator.py for
the LLM-based communication/clarity scores).

Also contains helper functions used to aggregate scores across a full
interview session for the final report and dashboard charts.
"""

import re

FILLER_WORDS = [
    "um", "uh", "like", "you know", "basically", "actually", "sort of",
    "kind of", "i mean", "literally", "so yeah", "stuff like that",
]

VAGUE_PHRASES = [
    "i think", "maybe", "not sure", "something like", "i guess",
    "probably", "kind of like",
]


def analyze_communication(text: str) -> dict:
    """Returns purely measured text statistics. Safe on empty input."""
    text = text or ""
    stripped = text.strip()

    if not stripped:
        return {
            "word_count": 0, "sentence_count": 0, "avg_sentence_length": 0.0,
            "filler_word_count": 0, "vague_phrase_count": 0,
            "unique_word_ratio": 0.0, "contains_example": False,
            "estimated_length_category": "No answer",
        }

    words = re.findall(r"[a-zA-Z']+", stripped.lower())
    word_count = len(words)

    sentences = re.split(r"[.!?]+", stripped)
    sentences = [s for s in sentences if s.strip()]
    sentence_count = max(1, len(sentences))

    avg_sentence_length = round(word_count / sentence_count, 1)

    lower_text = stripped.lower()
    filler_count = sum(lower_text.count(fw) for fw in FILLER_WORDS)
    vague_count = sum(lower_text.count(vp) for vp in VAGUE_PHRASES)

    unique_words = len(set(words))
    unique_word_ratio = round(unique_words / word_count, 2) if word_count else 0.0

    contains_example = bool(re.search(r"\bfor example\b|\bfor instance\b|\be\.g\.\b|\bsuch as\b|\bin my project\b|\bin my experience\b", lower_text))

    if word_count < 15:
        length_category = "Very short"
    elif word_count < 40:
        length_category = "Short"
    elif word_count < 120:
        length_category = "Well-balanced"
    elif word_count < 250:
        length_category = "Long"
    else:
        length_category = "Overly long"

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "filler_word_count": filler_count,
        "vague_phrase_count": vague_count,
        "unique_word_ratio": unique_word_ratio,
        "contains_example": contains_example,
        "estimated_length_category": length_category,
    }


def aggregate_session_scores(qa_history: list) -> dict:
    """Aggregates per-answer evaluation scores into overall session-level
    category scores for the final report and dashboard."""
    if not qa_history:
        return {
            "overall_score": 0, "technical_score": 0, "communication_score": 0,
            "problem_solving_score": 0, "confidence_score": 0,
        }

    evaluated = [qa for qa in qa_history if qa.get("evaluation")]
    if not evaluated:
        return {
            "overall_score": 0, "technical_score": 0, "communication_score": 0,
            "problem_solving_score": 0, "confidence_score": 0,
        }

    def avg(key):
        vals = [qa["evaluation"].get(key, 0) for qa in evaluated]
        return round(sum(vals) / len(vals), 1) if vals else 0

    technical_score = avg("technical_correctness")
    communication_score = avg("communication")
    completeness = avg("completeness")
    relevance = avg("relevance")
    confidence_score = avg("confidence_indicator")
    overall_score = avg("overall_score")
    # Problem solving approximated as blend of completeness + relevance +
    # technical correctness -- explainable proxy rather than a separate LLM call.
    problem_solving_score = round((completeness + relevance + technical_score) / 3, 1)

    return {
        "overall_score": overall_score,
        "technical_score": technical_score,
        "communication_score": communication_score,
        "problem_solving_score": problem_solving_score,
        "confidence_score": confidence_score,
    }


def aggregate_strengths_weaknesses(qa_history: list):
    strengths, weaknesses = [], []
    for qa in qa_history:
        ev = qa.get("evaluation", {})
        strengths.extend(ev.get("strengths", []))
        weaknesses.extend(ev.get("weaknesses", []))
    # De-duplicate while preserving order
    seen_s, seen_w = set(), set()
    unique_strengths = [s for s in strengths if not (s in seen_s or seen_s.add(s))]
    unique_weaknesses = [w for w in weaknesses if not (w in seen_w or seen_w.add(w))]
    return unique_strengths[:10], unique_weaknesses[:10]
