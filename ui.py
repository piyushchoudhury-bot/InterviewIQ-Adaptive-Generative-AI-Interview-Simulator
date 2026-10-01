"""
ui.py
-----
Custom CSS and reusable UI components for InterviewIQ, designed to look like
a modern SaaS product rather than a default Streamlit app. Implements a
horizontal TOP navigation bar (using st.columns + buttons) instead of the
Streamlit sidebar, per the project's core UI requirement.
"""

import streamlit as st

PAGES = ["Dashboard", "Resume & JD", "Interview Setup", "Interview", "Evaluation", "Skill Gap", "History"]

CUSTOM_CSS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}

    .block-container {
        padding-top: 1rem;
        max-width: 1200px;
    }

    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'Inter', -apple-system, sans-serif;
    }

    .iq-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(90deg, #4F46E5 0%, #6366F1 100%);
        padding: 14px 28px;
        border-radius: 16px;
        margin-bottom: 18px;
        box-shadow: 0 4px 16px rgba(79, 70, 229, 0.25);
    }
    .iq-navbar-title {
        color: white;
        font-size: 22px;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .iq-navbar-subtitle {
        color: #E0E7FF;
        font-size: 12px;
        font-weight: 400;
        margin-top: -2px;
    }

    div[data-testid="column"] div.stButton > button {
        width: 100%;
        border-radius: 10px;
        border: 1px solid #E5E7EB;
        background-color: #FFFFFF;
        color: #374151;
        font-weight: 600;
        padding: 0.5rem 0.4rem;
        transition: all 0.15s ease-in-out;
    }
    div[data-testid="column"] div.stButton > button:hover {
        border-color: #6366F1;
        color: #4F46E5;
        background-color: #EEF2FF;
    }
    .iq-nav-active button {
        background-color: #4F46E5 !important;
        color: white !important;
        border-color: #4F46E5 !important;
    }

    .iq-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 20px 22px;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
        border: 1px solid #F1F5F9;
        margin-bottom: 16px;
    }
    .iq-metric-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
        border: 1px solid #F1F5F9;
    }
    .iq-metric-value {
        font-size: 30px;
        font-weight: 800;
        color: #4F46E5;
        margin: 4px 0 0 0;
    }
    .iq-metric-label {
        font-size: 12.5px;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    .iq-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        margin-right: 6px;
    }
    .iq-badge-green { background: #DCFCE7; color: #166534; }
    .iq-badge-yellow { background: #FEF9C3; color: #854D0E; }
    .iq-badge-red { background: #FEE2E2; color: #991B1B; }
    .iq-badge-indigo { background: #E0E7FF; color: #3730A3; }
    .iq-badge-gray { background: #F1F5F9; color: #475569; }

    .iq-question-box {
        background: linear-gradient(135deg, #EEF2FF 0%, #F5F3FF 100%);
        border-left: 5px solid #6366F1;
        border-radius: 12px;
        padding: 18px 22px;
        font-size: 17px;
        font-weight: 500;
        color: #1E1B4B;
        margin-bottom: 14px;
    }

    .iq-section-title {
        font-size: 20px;
        font-weight: 800;
        color: #1E1B4B;
        margin: 6px 0 10px 0;
    }
    .iq-muted {
        color: #64748B;
        font-size: 13.5px;
    }
    .iq-demo-banner {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        color: #92400E;
        padding: 8px 16px;
        border-radius: 10px;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 12px;
    }
</style>
"""


def inject_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_navbar():
    """Renders the branded header and a horizontal top navigation bar built
    from st.columns + st.button (NOT st.sidebar)."""
    st.markdown(
        """
        <div class="iq-navbar">
            <div>
                <div class="iq-navbar-title">🎯 InterviewIQ</div>
                <div class="iq-navbar-subtitle">Adaptive Multimodal Generative AI Interview Simulator</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"

    cols = st.columns(len(PAGES))
    for col, page_name in zip(cols, PAGES):
        with col:
            is_active = st.session_state.page == page_name
            if is_active:
                st.markdown('<div class="iq-nav-active">', unsafe_allow_html=True)
            if st.button(page_name, key=f"nav_{page_name}", use_container_width=True):
                st.session_state.page = page_name
                st.rerun()
            if is_active:
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)


def metric_card(label: str, value, suffix: str = ""):
    st.markdown(
        f"""
        <div class="iq-metric-card">
            <div class="iq-metric-label">{label}</div>
            <div class="iq-metric-value">{value}{suffix}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge(text: str, kind: str = "gray"):
    return f'<span class="iq-badge iq-badge-{kind}">{text}</span>'


def score_badge_kind(score) -> str:
    try:
        score = float(score)
    except (TypeError, ValueError):
        return "gray"
    if score >= 75:
        return "green"
    if score >= 50:
        return "yellow"
    return "red"


def card_start():
    st.markdown('<div class="iq-card">', unsafe_allow_html=True)


def card_end():
    st.markdown("</div>", unsafe_allow_html=True)


def section_title(text: str):
    st.markdown(f'<div class="iq-section-title">{text}</div>', unsafe_allow_html=True)


def demo_banner(message: str):
    st.markdown(f'<div class="iq-demo-banner">🧪 {message}</div>', unsafe_allow_html=True)
