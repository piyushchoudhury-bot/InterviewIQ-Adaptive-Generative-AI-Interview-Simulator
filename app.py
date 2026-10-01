"""
InterviewIQ - Adaptive Multimodal Generative AI System for Personalized
Interview Simulation, Knowledge-Grounded Evaluation, and Skill-Gap Analysis.

Run with:  streamlit run app.py

Architecture (see README.md for full explanation):
Resume + JD -> Document Processing -> Resume/JD Analysis -> Skill Extraction
-> Interview Configuration -> Adaptive Interview Engine -> Candidate Answer
-> RAG Knowledge Retrieval -> LLM Evaluation -> Performance Metrics
-> Skill Gap Analysis -> Personalized Learning Plan -> History + Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src import (
    ui, database, document_loader, resume_analyzer, jd_analyzer,
    adaptive_interviewer, evaluator, skill_gap, analytics, report_generator,
    llm, rag,
)
from src.sample_data import SAMPLE_RESUME_TEXT, SAMPLE_JD_TEXT

st.set_page_config(page_title="InterviewIQ", page_icon="🎯", layout="wide")

# --------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# --------------------------------------------------------------------------
DEFAULTS = {
    "page": "Dashboard",
    "demo_mode": True,  # default ON so the app is demonstrable immediately
    "resume_text": "",
    "resume_data": {},
    "jd_text": "",
    "jd_data": {},
    "match_result": {},
    "interview_config": {},
    "interview_active": False,
    "interview_finished": False,
    "current_question": None,
    "question_index": 0,
    "qa_history": [],
    "skill_state": {},
    "current_difficulty": "medium",
    "final_report": None,
    "last_session_id": None,
    "viewing_history_id": None,
}
for key, default_value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

database.init_db()
ui.inject_css()
ui.render_navbar()

# --------------------------------------------------------------------------
# GLOBAL SETTINGS ROW (Demo Mode toggle) -- deliberately NOT in st.sidebar
# Kept deliberately quiet: only speaks up when something needs attention.
# --------------------------------------------------------------------------
with st.expander("⚙️ Settings", expanded=False):
    st.session_state.demo_mode = st.toggle(
        "Demo Mode", value=st.session_state.demo_mode,
        help="When ON, the app uses built-in sample data and does not call the Gemini API. Turn OFF to use your own resume/JD with live Gemini-powered questions and evaluation.",
    )
    if not st.session_state.demo_mode and not llm.is_available():
        st.warning(
            "No working Gemini API key was found, so live questions/evaluation won't work. "
            "Either turn Demo Mode back on, or add GEMINI_API_KEY to your .env file and restart the app."
        )
    if not st.session_state.demo_mode:
        rag_ready = rag.is_ready() or rag.build_index()
        if not rag_ready:
            st.caption(f"Note: knowledge-grounded evaluation is unavailable ({rag.get_build_error()}).")

PAGE = st.session_state.page


# ==========================================================================
# PAGE 1: DASHBOARD
# ==========================================================================
def page_dashboard():
    ui.section_title("Welcome to InterviewIQ")
    st.markdown(
        '<p class="iq-muted">Simulate personalized, adaptive interviews grounded in real knowledge, '
        "and get a data-driven skill-gap analysis with a 7-day learning plan.</p>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🚀 Start New Interview", use_container_width=True, type="primary"):
            st.session_state.page = "Interview Setup"
            st.rerun()
    with c2:
        if st.button("📄 Analyze Resume & JD", use_container_width=True):
            st.session_state.page = "Resume & JD"
            st.rerun()

    st.write("")
    sessions = database.get_all_sessions()

    ui.section_title("Performance Overview")
    if not sessions:
        st.markdown(
            '<div class="iq-card"><span class="iq-muted">No interviews completed yet. '
            "Start a new interview to see your performance metrics here.</span></div>",
            unsafe_allow_html=True,
        )
    else:
        latest = sessions[0]
        cols = st.columns(4)
        with cols[0]:
            ui.metric_card("Overall Score", round(latest["overall_score"], 1), "/100")
        with cols[1]:
            ui.metric_card("Technical Score", round(latest["technical_score"], 1), "/100")
        with cols[2]:
            ui.metric_card("Communication", round(latest["communication_score"], 1), "/100")
        with cols[3]:
            ui.metric_card("Confidence", round(latest["confidence_score"], 1), "/100")

        st.write("")
        readiness = latest["overall_score"]
        readiness_label = "Strong" if readiness >= 75 else ("Developing" if readiness >= 50 else "Needs Work")
        readiness_kind = ui.score_badge_kind(readiness)
        ui.card_start()
        st.markdown(f"**Skill Readiness Indicator:** {ui.badge(readiness_label, readiness_kind)}", unsafe_allow_html=True)
        st.progress(min(1.0, max(0.0, readiness / 100)))
        try:
            import json as _json
            gaps = _json.loads(latest.get("skill_gaps_json") or "{}")
            missing = gaps.get("missing_skills", [])
            if missing:
                st.markdown("**Recent Skill Gaps:** " + " ".join(ui.badge(m["skill"], "red") for m in missing[:6]), unsafe_allow_html=True)
        except Exception:
            pass
        ui.card_end()

        if len(sessions) > 1:
            ui.section_title("Performance Trend")
            df = pd.DataFrame(sessions[::-1])
            df["session_no"] = range(1, len(df) + 1)
            fig = px.line(
                df, x="session_no", y=["overall_score", "technical_score", "communication_score"],
                markers=True, labels={"value": "Score", "session_no": "Session #", "variable": "Metric"},
            )
            fig.update_layout(height=350, legend_title_text="", margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

        ui.section_title("Recent Interview Sessions")
        for s in sessions[:5]:
            ui.card_start()
            cols = st.columns([3, 1, 1, 1])
            cols[0].markdown(f"**{s['interview_type']} Interview** &nbsp;·&nbsp; {s['difficulty']} &nbsp;·&nbsp; {s['created_at'][:16]}")
            cols[1].markdown(f"Score: **{round(s['overall_score'],1)}**")
            cols[2].markdown(f"Qs: {s['questions_answered']}/{s['num_questions']}")
            if cols[3].button("View", key=f"dash_view_{s['id']}"):
                st.session_state.viewing_history_id = s["id"]
                st.session_state.page = "History"
                st.rerun()
            ui.card_end()


# ==========================================================================
# PAGE 2: RESUME & JOB DESCRIPTION
# ==========================================================================
def page_resume_jd():
    ui.section_title("Resume & Job Description Analysis")

    if st.session_state.demo_mode:
        ui.demo_banner("Demo Mode is ON — sample resume & JD data will be used.")

    col1, col2 = st.columns(2)
    with col1:
        ui.card_start()
        st.markdown("**Upload Resume** (PDF / DOCX / TXT)")
        uploaded_resume = st.file_uploader("Resume file", type=["pdf", "docx", "txt"], label_visibility="collapsed")
        ui.card_end()
    with col2:
        ui.card_start()
        st.markdown("**Paste Job Description**")
        jd_input = st.text_area("Job description text", height=180, label_visibility="collapsed",
                                 placeholder="Paste the full job description here...")
        ui.card_end()

    analyze_clicked = st.button("🔍 Analyze Resume & JD", type="primary")

    if analyze_clicked:
        if st.session_state.demo_mode:
            st.session_state.resume_text = SAMPLE_RESUME_TEXT
            st.session_state.jd_text = SAMPLE_JD_TEXT
        else:
            resume_text, err = ("", None)
            if uploaded_resume is not None:
                resume_text, err = document_loader.extract_text(uploaded_resume)
            if err:
                st.error(err)
            st.session_state.resume_text = resume_text
            st.session_state.jd_text = jd_input or ""

        if not st.session_state.resume_text and not st.session_state.demo_mode:
            st.warning("Please upload a resume file before analyzing (or enable Demo Mode).")
        else:
            with st.spinner("Extracting resume details, analyzing JD, and computing match score..."):
                st.session_state.resume_data = resume_analyzer.analyze_resume(
                    st.session_state.resume_text, demo_mode=st.session_state.demo_mode
                )
                st.session_state.jd_data = jd_analyzer.analyze_jd(
                    st.session_state.jd_text, demo_mode=st.session_state.demo_mode
                )
                st.session_state.match_result = jd_analyzer.compute_match(
                    st.session_state.resume_data, st.session_state.jd_data,
                    st.session_state.resume_text, st.session_state.jd_text,
                )
            st.success("Analysis complete!")

    if st.session_state.resume_data:
        rd = st.session_state.resume_data
        jd = st.session_state.jd_data
        match = st.session_state.match_result

        if not st.session_state.demo_mode and (rd.get("_llm_error") or jd.get("_llm_error")):
            st.warning(
                "⚠️ Gemini extraction wasn't used for at least one document, so a simpler "
                "backup extractor filled in the details below (it may be less complete). "
                f"Reason: {rd.get('_llm_error') or jd.get('_llm_error')}\n\n"
                "Check that GEMINI_API_KEY is set correctly in your .env file and restart the app, "
                "or turn on Demo Mode above."
            )

        ui.section_title("Resume Summary")
        ui.card_start()
        st.markdown(f"**Candidate:** {rd.get('name', 'N/A')}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Education**")
            for e in rd.get("education", []) or ["Not specified"]:
                st.markdown(f"- {e}")
            st.markdown("**Certifications**")
            for c in rd.get("certifications", []) or ["None listed"]:
                st.markdown(f"- {c}")
        with c2:
            st.markdown("**Experience**")
            for e in rd.get("experience", []) or ["Not specified"]:
                st.markdown(f"- {e}")
            st.markdown("**Projects**")
            for p in rd.get("projects", []) or ["Not specified"]:
                st.markdown(f"- {p}")
        st.markdown("**Technical Skills:** " + " ".join(ui.badge(s, "indigo") for s in rd.get("technical_skills", [])), unsafe_allow_html=True)
        st.markdown("**Soft Skills:** " + " ".join(ui.badge(s, "gray") for s in rd.get("soft_skills", [])), unsafe_allow_html=True)
        ui.card_end()

        ui.section_title("Job Description Summary")
        ui.card_start()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Required Skills:** " + " ".join(ui.badge(s, "red") for s in jd.get("required_skills", [])), unsafe_allow_html=True)
            st.markdown("**Preferred Skills:** " + " ".join(ui.badge(s, "yellow") for s in jd.get("preferred_skills", [])), unsafe_allow_html=True)
        with c2:
            st.markdown(f"**Experience Requirement:** {jd.get('experience_requirement', 'Not specified')}")
            st.markdown("**Qualifications**")
            for q in jd.get("qualifications", []) or ["Not specified"]:
                st.markdown(f"- {q}")
        st.markdown("**Responsibilities**")
        for r in jd.get("responsibilities", []) or ["Not specified"]:
            st.markdown(f"- {r}")
        ui.card_end()

        if match:
            ui.section_title("Resume ↔ JD Match")
            st.caption("⚠️ This is an application-generated compatibility score for practice purposes, NOT an official ATS score.")
            ui.card_start()
            cols = st.columns(3)
            with cols[0]:
                ui.metric_card("Overall Match", match["overall_match"], "%")
            with cols[1]:
                ui.metric_card("Semantic Similarity", match["semantic_similarity"], "%")
            with cols[2]:
                ui.metric_card("Skill Coverage", match["skill_coverage"], "%")
            if match.get("used_fallback_similarity"):
                st.caption("Note: semantic embedding model unavailable — similarity estimated via keyword overlap instead.")
            st.write("")
            st.markdown("**✅ Matching Skills:** " + (" ".join(ui.badge(s, "green") for s in match["matching_skills"]) or "None"), unsafe_allow_html=True)
            st.markdown("**❌ Missing Required Skills:** " + (" ".join(ui.badge(s, "red") for s in match["missing_required_skills"]) or "None"), unsafe_allow_html=True)
            st.markdown("**➕ Missing Preferred Skills:** " + (" ".join(ui.badge(s, "yellow") for s in match["missing_preferred_skills"]) or "None"), unsafe_allow_html=True)
            st.markdown(f"**Experience Alignment:** {match['experience_alignment']}")
            ui.card_end()

        if st.button("Proceed to Interview Setup →", type="primary"):
            st.session_state.page = "Interview Setup"
            st.rerun()


# ==========================================================================
# PAGE 3: INTERVIEW SETUP
# ==========================================================================
def page_interview_setup():
    ui.section_title("Interview Setup")

    if not st.session_state.resume_data:
        st.warning("Tip: Analyze your Resume & JD first so focus areas can be auto-suggested. You can also proceed without it.")

    ui.card_start()
    col1, col2 = st.columns(2)
    with col1:
        interview_type = st.selectbox("Interview Type", ["Technical", "HR", "Behavioral", "Mixed"])
        difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
    with col2:
        duration = st.selectbox("Interview Duration", ["10 minutes", "20 minutes", "30 minutes", "Custom"])
        if duration == "Custom":
            duration = f"{st.number_input('Custom duration (minutes)', min_value=5, max_value=120, value=15)} minutes"

        num_q_choice = st.selectbox("Number of Questions", ["5", "10", "15", "Custom"])
        if num_q_choice == "Custom":
            num_questions = st.number_input("Custom number of questions", min_value=1, max_value=30, value=7)
        else:
            num_questions = int(num_q_choice)

    st.markdown("**Focus Areas**")
    auto_focus = st.checkbox("Use Resume + JD to automatically determine focus areas", value=True)

    resume_skills = st.session_state.resume_data.get("technical_skills", []) if st.session_state.resume_data else []
    jd_skills = (st.session_state.jd_data.get("required_skills", []) + st.session_state.jd_data.get("preferred_skills", [])) if st.session_state.jd_data else []
    default_pool = ["Python", "SQL", "Machine Learning", "Data Structures", "Data Engineering", "Cloud", "GenAI", "Statistics"]
    combined_pool = sorted(set([s for s in (resume_skills + jd_skills) if s in default_pool] or default_pool))
    all_options = sorted(set(default_pool + combined_pool))

    if auto_focus:
        focus_areas = combined_pool if combined_pool else default_pool[:4]
        st.multiselect("Auto-selected focus areas (from resume/JD)", options=all_options, default=focus_areas, disabled=True)
    else:
        focus_areas = st.multiselect("Select focus areas manually", options=all_options, default=default_pool[:4])
    ui.card_end()

    if st.button("▶️ START INTERVIEW", type="primary", use_container_width=True):
        st.session_state.interview_config = {
            "interview_type": interview_type,
            "difficulty": difficulty.lower(),
            "duration": duration,
            "num_questions": int(num_questions),
            "focus_areas": focus_areas,
        }
        st.session_state.qa_history = []
        st.session_state.skill_state = {}
        st.session_state.current_difficulty = difficulty.lower()
        st.session_state.question_index = 1
        st.session_state.interview_active = True
        st.session_state.interview_finished = False
        st.session_state.final_report = None

        context = _build_adaptive_context()
        with st.spinner("Generating your first question..."):
            st.session_state.current_question = adaptive_interviewer.generate_next_question(
                context, demo_mode=st.session_state.demo_mode
            )
        st.session_state.page = "Interview"
        st.rerun()


# ==========================================================================
# PAGE 4: INTERVIEW SIMULATOR
# ==========================================================================
def _build_adaptive_context():
    cfg = st.session_state.interview_config
    return {
        "interview_type": cfg.get("interview_type", "Mixed"),
        "difficulty": cfg.get("difficulty", "medium"),
        "current_difficulty": st.session_state.current_difficulty,
        "focus_areas": cfg.get("focus_areas", []),
        "resume_skills": st.session_state.resume_data.get("technical_skills", []) if st.session_state.resume_data else [],
        "jd_skills": (st.session_state.jd_data.get("required_skills", []) + st.session_state.jd_data.get("preferred_skills", [])) if st.session_state.jd_data else [],
        "qa_history": st.session_state.qa_history,
        "skill_state": st.session_state.skill_state,
        "question_number": st.session_state.question_index,
        "total_questions": cfg.get("num_questions", 5),
    }


def _finish_interview():
    scores = analytics.aggregate_session_scores(st.session_state.qa_history)
    strengths, weaknesses = analytics.aggregate_strengths_weaknesses(st.session_state.qa_history)
    resume_alignment = st.session_state.match_result.get("overall_match", 0) if st.session_state.match_result else 0

    with st.spinner("Analyzing skill gaps and building your personalized learning plan..."):
        sg_report = skill_gap.build_skill_gap_report(
            st.session_state.resume_data or {}, st.session_state.jd_data or {},
            st.session_state.skill_state, demo_mode=st.session_state.demo_mode,
        )
        plan = skill_gap.build_learning_plan(sg_report, demo_mode=st.session_state.demo_mode)

    report = {
        "candidate_name": (st.session_state.resume_data or {}).get("name", "Candidate"),
        "interview_type": st.session_state.interview_config.get("interview_type"),
        "difficulty": st.session_state.interview_config.get("difficulty"),
        "num_questions": st.session_state.interview_config.get("num_questions"),
        "questions_answered": len(st.session_state.qa_history),
        "overall_score": scores["overall_score"],
        "technical_score": scores["technical_score"],
        "communication_score": scores["communication_score"],
        "problem_solving_score": scores["problem_solving_score"],
        "confidence_score": scores["confidence_score"],
        "resume_alignment_score": resume_alignment,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "qa_history": st.session_state.qa_history,
        "skill_gap_report": sg_report,
        "learning_plan": plan,
        "resume_summary": st.session_state.resume_data,
        "jd_summary": st.session_state.jd_data,
    }
    session_id = database.save_session({**report, "skill_gaps": sg_report})
    st.session_state.last_session_id = session_id
    st.session_state.final_report = report
    st.session_state.interview_active = False
    st.session_state.interview_finished = True


def page_interview():
    if not st.session_state.interview_active:
        st.info("No interview in progress. Configure and start one from Interview Setup.")
        if st.button("Go to Interview Setup"):
            st.session_state.page = "Interview Setup"
            st.rerun()
        return

    cfg = st.session_state.interview_config
    total = cfg.get("num_questions", 5)
    idx = st.session_state.question_index
    q = st.session_state.current_question

    if st.session_state.demo_mode:
        ui.demo_banner("Demo Mode is ON — questions and evaluation are generated deterministically (no API calls).")

    ui.card_start()
    st.markdown(f"**INTERVIEW QUESTION {idx} / {total}**")
    st.progress(min(1.0, (idx - 1) / total))
    ui.card_end()

    if q:
        st.markdown(
            f'<div class="iq-question-box">{q["question"]}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Topic: {q.get('skill', 'General')} · Difficulty: {q.get('difficulty', 'medium').title()} · Type: {q.get('question_type', 'Technical')}")
        with st.expander("Why this question? (adaptive engine reasoning)"):
            st.write(q.get("reason_for_selection", "N/A"))

    answer = st.text_area("Your Answer", height=180, key=f"answer_input_{idx}", placeholder="Type your answer here...")

    st.caption("🎙️ Optional voice input: upload an audio file (feature depends on a connected speech-to-text service).")
    audio_file = st.file_uploader("Audio answer (optional)", type=["wav", "mp3", "m4a"], key=f"audio_{idx}", label_visibility="collapsed")
    if audio_file is not None:
        st.warning("Audio analysis is unavailable. Text-based evaluation is being used.")

    c1, c2, c3 = st.columns(3)
    submit_clicked = c1.button("✅ Submit Answer", type="primary", use_container_width=True)
    skip_clicked = c2.button("⏭️ Skip Question", use_container_width=True)
    end_clicked = c3.button("🛑 End Interview", use_container_width=True)

    if submit_clicked or skip_clicked:
        final_answer = "" if skip_clicked else (answer or "")
        with st.spinner("Evaluating your answer..."):
            evaluation = evaluator.evaluate_answer(
                question=q["question"], answer=final_answer, skill=q.get("skill", "General"),
                difficulty=q.get("difficulty", "medium"), expected_concepts=q.get("expected_concepts", []),
                demo_mode=st.session_state.demo_mode,
            )
        st.session_state.qa_history.append({
            "question": q["question"], "answer": final_answer if final_answer else "(Skipped)",
            "skill": q.get("skill", "General"), "difficulty": q.get("difficulty", "medium"),
            "question_type": q.get("question_type", "Technical"), "evaluation": evaluation,
        })
        st.session_state.skill_state = adaptive_interviewer.update_skill_state(
            st.session_state.skill_state, q.get("skill", "General"), evaluation.get("overall_score", 0)
        )
        # current_difficulty is re-derived adaptively inside generate_next_question
        # based on skill_state, so it is left as-is here.

        if idx >= total:
            _finish_interview()
            st.session_state.page = "Evaluation"
            st.rerun()
        else:
            st.session_state.question_index += 1
            context = _build_adaptive_context()
            with st.spinner("Generating your next question based on your performance..."):
                st.session_state.current_question = adaptive_interviewer.generate_next_question(
                    context, demo_mode=st.session_state.demo_mode
                )
            st.rerun()

    if end_clicked:
        if st.session_state.qa_history:
            _finish_interview()
            st.session_state.page = "Evaluation"
        else:
            st.session_state.interview_active = False
            st.session_state.page = "Dashboard"
        st.rerun()


# ==========================================================================
# PAGE 5: EVALUATION
# ==========================================================================
def page_evaluation():
    report = st.session_state.final_report
    if not report:
        st.info("No completed interview to evaluate yet. Complete an interview first.")
        if st.button("Go to Interview Setup"):
            st.session_state.page = "Interview Setup"
            st.rerun()
        return

    ui.section_title("Final Interview Evaluation")
    cols = st.columns(4)
    with cols[0]:
        ui.metric_card("Overall Score", report["overall_score"], "/100")
    with cols[1]:
        ui.metric_card("Technical", report["technical_score"], "/100")
    with cols[2]:
        ui.metric_card("Communication", report["communication_score"], "/100")
    with cols[3]:
        ui.metric_card("Confidence", report["confidence_score"], "/100")

    st.write("")
    ui.section_title("Category Breakdown")
    c1, c2 = st.columns(2)
    with c1:
        categories = ["Technical Knowledge", "Communication", "Problem Solving", "Answer Quality", "Resume Alignment", "Confidence"]
        values = [
            report["technical_score"], report["communication_score"], report["problem_solving_score"],
            report["overall_score"], report["resume_alignment_score"], report["confidence_score"],
        ]
        fig = go.Figure(data=go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]], fill="toself", line_color="#4F46E5"))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=380, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = go.Figure(go.Bar(x=values, y=categories, orientation="h", marker_color="#6366F1"))
        fig2.update_layout(xaxis_range=[0, 100], height=380, margin=dict(t=20, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        ui.card_start()
        st.markdown("**💪 Strengths**")
        for s in report["strengths"] or ["None recorded"]:
            st.markdown(f"- {s}")
        ui.card_end()
    with c2:
        ui.card_start()
        st.markdown("**⚠️ Weaknesses**")
        for w in report["weaknesses"] or ["None recorded"]:
            st.markdown(f"- {w}")
        ui.card_end()

    ui.section_title("Question-by-Question Detail")
    for i, qa in enumerate(report["qa_history"], 1):
        ev = qa["evaluation"]
        with st.expander(f"Q{i}: {qa['question']}  —  Score: {ev.get('overall_score', 0)}/100"):
            st.markdown(f"**Your answer:** {qa['answer']}")
            st.markdown(f"**Topic:** {qa['skill']} · **Difficulty:** {qa['difficulty']}")
            if ev.get("is_knowledge_grounded"):
                st.caption("✅ This evaluation was knowledge-grounded using the RAG pipeline.")
            else:
                st.caption("ℹ️ This evaluation was not knowledge-grounded (RAG unavailable or Demo Mode).")
            score_cols = st.columns(6)
            for col, key, label in zip(
                score_cols,
                ["technical_correctness", "completeness", "relevance", "clarity", "communication", "confidence_indicator"],
                ["Technical", "Completeness", "Relevance", "Clarity", "Communication", "Confidence"],
            ):
                col.metric(label, f"{ev.get(key, 0)}")
            comm = ev.get("communication_analysis", {})
            st.caption(
                f"Measured text stats — Words: {comm.get('word_count', 0)} · "
                f"Sentences: {comm.get('sentence_count', 0)} · "
                f"Avg sentence length: {comm.get('avg_sentence_length', 0)} · "
                f"Filler words: {comm.get('filler_word_count', 0)} · "
                f"Length: {comm.get('estimated_length_category', 'N/A')}"
            )
            if ev.get("missing_concepts"):
                st.markdown("**Missing concepts:** " + ", ".join(ev["missing_concepts"]))
            if ev.get("recommended_improvement"):
                st.markdown(f"**Improvement tip:** {ev['recommended_improvement']}")

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        pdf_bytes = report_generator.generate_interview_report(report)
        st.download_button("⬇️ Download Interview Report (PDF)", data=pdf_bytes,
                            file_name="InterviewIQ_Interview_Report.pdf", mime="application/pdf",
                            use_container_width=True)
    with c2:
        if st.button("View Skill Gap Analysis →", use_container_width=True, type="primary"):
            st.session_state.page = "Skill Gap"
            st.rerun()


# ==========================================================================
# PAGE 6: SKILL GAP
# ==========================================================================
def page_skill_gap():
    report = st.session_state.final_report
    if not report or not report.get("skill_gap_report"):
        st.info("No skill-gap analysis available yet. Complete an interview first.")
        if st.button("Go to Interview Setup"):
            st.session_state.page = "Interview Setup"
            st.rerun()
        return

    sg = report["skill_gap_report"]
    ui.section_title("Skill Gap Analysis")
    st.caption("Comparing Resume skills vs. JD requirements vs. skills actually demonstrated in your interview.")

    c1, c2, c3 = st.columns(3)
    with c1:
        ui.card_start()
        st.markdown("**🟢 Strong Skills**")
        for s in sg["strong_skills"] or [{"skill": "None yet", "evidence": ""}]:
            st.markdown(f"- **{s['skill']}**  \n<span class='iq-muted'>{s.get('evidence','')}</span>", unsafe_allow_html=True)
        ui.card_end()
    with c2:
        ui.card_start()
        st.markdown("**🟡 Needs Improvement**")
        for s in sg["needs_improvement"] or [{"skill": "None", "evidence": ""}]:
            st.markdown(f"- **{s['skill']}** {ui.badge(s.get('priority','Medium'), 'yellow')}  \n<span class='iq-muted'>{s.get('evidence','')}</span>", unsafe_allow_html=True)
        ui.card_end()
    with c3:
        ui.card_start()
        st.markdown("**🔴 Missing Skills**")
        for s in sg["missing_skills"] or [{"skill": "None", "evidence": ""}]:
            st.markdown(f"- **{s['skill']}** {ui.badge(s.get('priority','Medium'), 'red')}  \n<span class='iq-muted'>{s.get('evidence','')}</span>", unsafe_allow_html=True)
        ui.card_end()

    ui.section_title("Detailed Recommendations")
    for rec in sg.get("recommendations", []):
        ui.card_start()
        st.markdown(f"### {rec['skill']}  {ui.badge(rec.get('priority','Medium'), ui.score_badge_kind(100 if rec.get('priority')=='Low' else (40 if rec.get('priority')=='High' else 60)))}", unsafe_allow_html=True)
        st.markdown(f"**Why it matters:** {rec.get('why_it_matters', '')}")
        st.markdown(f"**Current evidence:** {rec.get('current_evidence', '')}")
        st.markdown(f"**Recommended topic:** {rec.get('recommended_topic', '')}")
        st.markdown(f"**Practice question:** {rec.get('practice_question', '')}")
        ui.card_end()

    ui.section_title("📅 7-Day Personalized Learning Plan")
    plan = report.get("learning_plan", [])
    tabs = st.tabs([f"Day {d['day']}" for d in plan])
    for tab, day_plan in zip(tabs, plan):
        with tab:
            st.markdown(f"#### {day_plan['topic']}")
            st.markdown("**Tasks:**")
            for t in day_plan["tasks"]:
                st.markdown(f"- {t}")
            st.markdown(f"**Practice question:** {day_plan['practice_question']}")

    st.write("")
    pdf_bytes = report_generator.generate_skill_gap_report(sg, plan, report.get("candidate_name", "Candidate"))
    st.download_button("⬇️ Download Skill Gap Report (PDF)", data=pdf_bytes,
                        file_name="InterviewIQ_SkillGap_Report.pdf", mime="application/pdf")


# ==========================================================================
# PAGE 7: HISTORY
# ==========================================================================
def page_history():
    ui.section_title("Interview History")
    sessions = database.get_all_sessions()

    if not sessions:
        st.info("No interview sessions recorded yet.")
        return

    if len(sessions) > 1:
        df = pd.DataFrame(sessions[::-1])
        df["session_no"] = range(1, len(df) + 1)
        fig = px.line(df, x="session_no", y="overall_score", markers=True,
                       labels={"overall_score": "Overall Score", "session_no": "Session #"})
        fig.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    for s in sessions:
        ui.card_start()
        cols = st.columns([3, 1, 1, 1, 1])
        cols[0].markdown(f"**{s['interview_type']} Interview** ({s['difficulty']}) — {s['created_at'][:16]}")
        cols[1].markdown(f"Overall: **{round(s['overall_score'],1)}**")
        cols[2].markdown(f"Tech: {round(s['technical_score'],1)}")
        cols[3].markdown(f"Comm: {round(s['communication_score'],1)}")
        if cols[4].button("Open", key=f"hist_open_{s['id']}"):
            st.session_state.viewing_history_id = s["id"] if st.session_state.viewing_history_id != s["id"] else None
        ui.card_end()

        if st.session_state.viewing_history_id == s["id"]:
            import json as _json
            with st.expander("Session detail", expanded=True):
                qa_history = _json.loads(s.get("qa_history_json") or "[]")
                for i, qa in enumerate(qa_history, 1):
                    ev = qa.get("evaluation", {})
                    st.markdown(f"**Q{i} [{qa.get('skill')}]:** {qa.get('question')}")
                    st.markdown(f"Answer: {qa.get('answer')}")
                    st.markdown(f"Score: {ev.get('overall_score', 'N/A')}/100")
                    st.markdown("---")
                gaps = _json.loads(s.get("skill_gaps_json") or "{}")
                if gaps:
                    st.markdown("**Missing skills at the time:** " + ", ".join(m["skill"] for m in gaps.get("missing_skills", [])) or "None")
                if st.button("Delete this session", key=f"del_{s['id']}"):
                    database.delete_session(s["id"])
                    st.session_state.viewing_history_id = None
                    st.rerun()


# --------------------------------------------------------------------------
# ROUTER
# --------------------------------------------------------------------------
ROUTES = {
    "Dashboard": page_dashboard,
    "Resume & JD": page_resume_jd,
    "Interview Setup": page_interview_setup,
    "Interview": page_interview,
    "Evaluation": page_evaluation,
    "Skill Gap": page_skill_gap,
    "History": page_history,
}

ROUTES.get(PAGE, page_dashboard)()
