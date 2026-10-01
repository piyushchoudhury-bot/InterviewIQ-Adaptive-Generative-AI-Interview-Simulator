"""
report_generator.py
--------------------
Generates downloadable PDF reports (Interview Report, Skill Gap Report)
using ReportLab. Returns raw PDF bytes so app.py can feed them directly into
st.download_button without ever writing to disk.
"""

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

styles = getSampleStyleSheet()
TITLE_STYLE = ParagraphStyle("TitleStyle", parent=styles["Title"], textColor=colors.HexColor("#4F46E5"))
H2_STYLE = ParagraphStyle("H2Style", parent=styles["Heading2"], textColor=colors.HexColor("#312E81"), spaceBefore=14, spaceAfter=6)
H3_STYLE = ParagraphStyle("H3Style", parent=styles["Heading3"], textColor=colors.HexColor("#4338CA"), spaceBefore=10, spaceAfter=4)
BODY_STYLE = ParagraphStyle("BodyStyle", parent=styles["BodyText"], spaceAfter=4, leading=14)
SMALL_STYLE = ParagraphStyle("SmallStyle", parent=styles["BodyText"], fontSize=8, textColor=colors.grey)


def _bullet_list(items):
    if not items:
        return Paragraph("None recorded.", BODY_STYLE)
    return ListFlowable(
        [ListItem(Paragraph(str(i), BODY_STYLE)) for i in items],
        bulletType="bullet",
    )


def _safe(text, default="N/A"):
    return text if text not in (None, "", []) else default


def generate_interview_report(report_data: dict) -> bytes:
    """
    report_data expected keys: candidate_name, interview_type, difficulty,
    num_questions, overall_score, technical_score, communication_score,
    problem_solving_score, confidence_score, resume_alignment_score,
    strengths, weaknesses, qa_history (list of dicts with question, answer,
    evaluation), skill_gap_report, learning_plan, resume_summary, jd_summary.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    story = []

    story.append(Paragraph("InterviewIQ - Interview Report", TITLE_STYLE))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%d %b %Y, %H:%M')}", SMALL_STYLE))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Candidate & Interview Configuration", H2_STYLE))
    config_table_data = [
        ["Candidate", _safe(report_data.get("candidate_name"))],
        ["Interview Type", _safe(report_data.get("interview_type"))],
        ["Difficulty", _safe(report_data.get("difficulty"))],
        ["Questions Answered", str(report_data.get("questions_answered", 0)) + " / " + str(report_data.get("num_questions", 0))],
    ]
    t = Table(config_table_data, colWidths=[5 * cm, 10 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF2FF")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C7D2FE")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Overall Performance", H2_STYLE))
    score_table_data = [
        ["Overall Score", "Technical", "Communication", "Problem Solving", "Confidence", "Resume Alignment"],
        [
            f"{report_data.get('overall_score', 0)}/100",
            f"{report_data.get('technical_score', 0)}/100",
            f"{report_data.get('communication_score', 0)}/100",
            f"{report_data.get('problem_solving_score', 0)}/100",
            f"{report_data.get('confidence_score', 0)}/100",
            f"{report_data.get('resume_alignment_score', 0)}/100",
        ],
    ]
    t2 = Table(score_table_data, colWidths=[2.5 * cm] * 6)
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F46E5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C7D2FE")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(t2)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Strengths", H3_STYLE))
    story.append(_bullet_list(report_data.get("strengths", [])))
    story.append(Paragraph("Weaknesses", H3_STYLE))
    story.append(_bullet_list(report_data.get("weaknesses", [])))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Question-by-Question Breakdown", H2_STYLE))
    for i, qa in enumerate(report_data.get("qa_history", []), 1):
        ev = qa.get("evaluation", {})
        story.append(Paragraph(f"Q{i} [{qa.get('skill', 'N/A')} | {qa.get('difficulty', 'N/A')}]: {qa.get('question', '')}", H3_STYLE))
        story.append(Paragraph(f"<b>Answer:</b> {_safe(qa.get('answer'), '(Skipped)')}", BODY_STYLE))
        story.append(Paragraph(
            f"<b>Score:</b> {ev.get('overall_score', 'N/A')}/100 &nbsp;|&nbsp; "
            f"Technical: {ev.get('technical_correctness', 'N/A')} &nbsp;|&nbsp; "
            f"Completeness: {ev.get('completeness', 'N/A')} &nbsp;|&nbsp; "
            f"Clarity: {ev.get('clarity', 'N/A')}",
            BODY_STYLE,
        ))
        if ev.get("recommended_improvement"):
            story.append(Paragraph(f"<i>Improvement tip:</i> {ev.get('recommended_improvement')}", BODY_STYLE))
        story.append(Spacer(1, 6))

    if report_data.get("skill_gap_report"):
        sg = report_data["skill_gap_report"]
        story.append(Paragraph("Skill Gap Summary", H2_STYLE))
        story.append(Paragraph("Strong Skills", H3_STYLE))
        story.append(_bullet_list([s["skill"] for s in sg.get("strong_skills", [])]))
        story.append(Paragraph("Needs Improvement", H3_STYLE))
        story.append(_bullet_list([s["skill"] for s in sg.get("needs_improvement", [])]))
        story.append(Paragraph("Missing Skills", H3_STYLE))
        story.append(_bullet_list([s["skill"] for s in sg.get("missing_skills", [])]))

    if report_data.get("learning_plan"):
        story.append(Paragraph("7-Day Personalized Learning Plan", H2_STYLE))
        for day_plan in report_data["learning_plan"]:
            story.append(Paragraph(f"Day {day_plan['day']}: {day_plan['topic']}", H3_STYLE))
            story.append(_bullet_list(day_plan.get("tasks", [])))

    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "Note: Scores are generated by an AI-assisted evaluation system for practice purposes only "
        "and do not represent an official assessment by any employer or certification body.",
        SMALL_STYLE,
    ))

    doc.build(story)
    return buffer.getvalue()


def generate_skill_gap_report(skill_gap_report: dict, learning_plan: list, candidate_name: str = "Candidate") -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    story = []

    story.append(Paragraph("InterviewIQ - Skill Gap Report", TITLE_STYLE))
    story.append(Paragraph(f"Candidate: {candidate_name} | Generated on {datetime.now().strftime('%d %b %Y, %H:%M')}", SMALL_STYLE))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Strong Skills", H2_STYLE))
    story.append(_bullet_list([f"{s['skill']} -- {s['evidence']}" for s in skill_gap_report.get("strong_skills", [])]))

    story.append(Paragraph("Needs Improvement", H2_STYLE))
    story.append(_bullet_list([f"{s['skill']} ({s.get('priority', 'Medium')} priority) -- {s['evidence']}" for s in skill_gap_report.get("needs_improvement", [])]))

    story.append(Paragraph("Missing Skills", H2_STYLE))
    story.append(_bullet_list([f"{s['skill']} ({s.get('priority', 'Medium')} priority) -- {s['evidence']}" for s in skill_gap_report.get("missing_skills", [])]))

    story.append(Paragraph("Detailed Recommendations", H2_STYLE))
    for rec in skill_gap_report.get("recommendations", []):
        story.append(Paragraph(f"{rec['skill']} ({rec.get('priority', 'Medium')} priority)", H3_STYLE))
        story.append(Paragraph(f"<b>Why it matters:</b> {rec.get('why_it_matters', '')}", BODY_STYLE))
        story.append(Paragraph(f"<b>Current evidence:</b> {rec.get('current_evidence', '')}", BODY_STYLE))
        story.append(Paragraph(f"<b>Recommended topic:</b> {rec.get('recommended_topic', '')}", BODY_STYLE))
        story.append(Paragraph(f"<b>Practice question:</b> {rec.get('practice_question', '')}", BODY_STYLE))
        story.append(Spacer(1, 6))

    if learning_plan:
        story.append(Paragraph("7-Day Personalized Learning Plan", H2_STYLE))
        for day_plan in learning_plan:
            story.append(Paragraph(f"Day {day_plan['day']}: {day_plan['topic']}", H3_STYLE))
            story.append(_bullet_list(day_plan.get("tasks", [])))
            story.append(Paragraph(f"<i>Practice:</i> {day_plan.get('practice_question', '')}", BODY_STYLE))

    doc.build(story)
    return buffer.getvalue()
