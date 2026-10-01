"""
evaluator.py
------------
Evaluates a single candidate answer using:
  1. RAG-retrieved reference material (src/rag.py) as grounding context.
  2. Gemini to score technical correctness, completeness, relevance,
     clarity, communication, and confidence, plus extract strengths/
     weaknesses/missing concepts.
  3. Deterministic communication analytics (word count, filler words, etc.)
     computed directly from the text -- NOT inferred by the LLM.

If the LLM call fails or returns malformed JSON, a rule-based fallback
evaluation is produced so the interview can always continue.
"""

import re

from src import llm, rag
from src.analytics import analyze_communication

FILLER_WORDS = ["um", "uh", "like", "you know", "basically", "actually", "sort of", "kind of", "i mean"]


def _rule_based_fallback_score(answer: str, expected_concepts: list) -> dict:
    """A simple, transparent heuristic used only when the LLM is unavailable.
    Rewards answer length and presence of expected-concept keywords."""
    answer_lower = answer.lower().strip()
    word_count = len(answer_lower.split())

    if not answer_lower or word_count < 3:
        return {
            "technical_correctness": 0, "completeness": 0, "relevance": 0,
            "clarity": 0, "communication": 0, "confidence_indicator": 0,
            "overall_score": 0,
            "strengths": [],
            "weaknesses": ["No substantive answer was provided."],
            "missing_concepts": expected_concepts,
            "recommended_improvement": "Attempt to answer with at least a few sentences covering the key concepts.",
            "is_knowledge_grounded": False,
        }

    concept_hits = sum(1 for c in expected_concepts if c.lower() in answer_lower)
    concept_ratio = (concept_hits / len(expected_concepts)) if expected_concepts else 0.5

    length_score = min(100, (word_count / 60) * 100)
    concept_score = concept_ratio * 100

    technical = round(0.7 * concept_score + 0.3 * length_score, 1)
    completeness = round(concept_score, 1)
    relevance = round(min(100, length_score + 10), 1)
    clarity = round(min(100, 60 + (10 if word_count > 15 else 0)), 1)
    communication = clarity
    confidence = round(min(100, 50 + concept_score / 2), 1)
    overall = round((technical + completeness + relevance + clarity + communication) / 5, 1)

    missing = [c for c in expected_concepts if c.lower() not in answer_lower]
    strengths = ["Answer addresses the question directly."] if concept_ratio > 0.3 else []
    weaknesses = [] if concept_ratio > 0.5 else ["Answer could cover more of the expected key concepts."]

    return {
        "technical_correctness": technical, "completeness": completeness, "relevance": relevance,
        "clarity": clarity, "communication": communication, "confidence_indicator": confidence,
        "overall_score": overall,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "missing_concepts": missing,
        "recommended_improvement": "Try to explicitly mention: " + ", ".join(missing) if missing else "Good coverage of expected concepts.",
        "is_knowledge_grounded": False,
    }


def evaluate_answer(question: str, answer: str, skill: str, difficulty: str,
                     expected_concepts: list, demo_mode: bool = False) -> dict:
    """
    Returns a dict with keys:
        technical_correctness, completeness, relevance, clarity, communication,
        confidence_indicator, overall_score, strengths, weaknesses,
        missing_concepts, recommended_improvement, is_knowledge_grounded,
        retrieved_context, communication_analysis
    """
    communication_analysis = analyze_communication(answer)

    retrieved_context = []
    grounded = False
    if not demo_mode:
        try:
            retrieved_context = rag.retrieve_context(f"{skill}: {question}", top_k=3)
            grounded = len(retrieved_context) > 0
        except Exception:
            retrieved_context = []
            grounded = False

    if demo_mode or not llm.is_available():
        result = _rule_based_fallback_score(answer, expected_concepts)
        result["is_knowledge_grounded"] = grounded
        result["retrieved_context"] = retrieved_context
        result["communication_analysis"] = communication_analysis
        return result

    context_text = rag.format_context_for_prompt(retrieved_context) if retrieved_context else "No reference material retrieved."

    prompt = f"""
You are evaluating a candidate's answer in a {difficulty}-difficulty interview
question about "{skill}".

QUESTION: {question}

CANDIDATE ANSWER: \"\"\"{answer.strip() or "(No answer provided / question was skipped)"}\"\"\"

EXPECTED KEY CONCEPTS: {', '.join(expected_concepts) if expected_concepts else 'Not specified'}

REFERENCE KNOWLEDGE (retrieved from a knowledge base, use ONLY to check
factual/technical accuracy -- do not assume the candidate has seen this text):
{context_text}

Evaluate the candidate's answer. Score each dimension from 0 to 100:
- technical_correctness: factual/technical accuracy compared to the reference knowledge
- completeness: how many expected concepts were covered
- relevance: how directly the answer addresses the question asked
- clarity: how clearly the answer is structured and explained
- communication: overall communication quality (structure, conciseness, examples)
- confidence_indicator: apparent confidence conveyed by the answer's tone and certainty of language (this is inferred from text only, not a measured vocal/behavioral signal)
- overall_score: your holistic 0-100 score weighing all of the above

Also extract:
- strengths: list of specific things the candidate did well (empty list if none)
- weaknesses: list of specific gaps or errors (empty list if none)
- missing_concepts: list of expected concepts NOT mentioned in the answer
- recommended_improvement: one or two sentences of specific, actionable advice

If the answer contains a claim that contradicts the reference knowledge,
note it as a weakness. If no answer was provided (skipped), score all
dimensions near 0 and note that no answer was given.

Respond with ONLY a JSON object with these exact keys: technical_correctness,
completeness, relevance, clarity, communication, confidence_indicator,
overall_score, strengths, weaknesses, missing_concepts, recommended_improvement.
"""
    result, error = llm.generate_json(prompt, temperature=0.3)
    if result is None or not isinstance(result, dict):
        fallback = _rule_based_fallback_score(answer, expected_concepts)
        fallback["is_knowledge_grounded"] = grounded
        fallback["retrieved_context"] = retrieved_context
        fallback["communication_analysis"] = communication_analysis
        fallback["_llm_error"] = error
        return fallback

    numeric_keys = ["technical_correctness", "completeness", "relevance", "clarity", "communication", "confidence_indicator", "overall_score"]
    for key in numeric_keys:
        try:
            result[key] = max(0, min(100, float(result.get(key, 0))))
        except (TypeError, ValueError):
            result[key] = 0

    for key in ["strengths", "weaknesses", "missing_concepts"]:
        if key not in result or not isinstance(result[key], list):
            result[key] = []

    result.setdefault("recommended_improvement", "")
    result["is_knowledge_grounded"] = grounded
    result["retrieved_context"] = retrieved_context
    result["communication_analysis"] = communication_analysis
    return result
