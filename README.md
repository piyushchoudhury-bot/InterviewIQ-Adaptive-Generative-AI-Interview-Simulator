# InterviewIQ

### An Adaptive Multimodal Generative AI System for Personalized Interview Simulation, Knowledge-Grounded Evaluation, and Skill-Gap Analysis

InterviewIQ is a Streamlit web application that simulates a personalized technical/HR interview based on a candidate's resume and a target job description. It asks adaptive questions (each one chosen based on how the candidate performed so far), evaluates answers using an LLM grounded in a retrieval-augmented (RAG) knowledge base, and produces a skill-gap report with a 7-day personalized learning plan.

Built as a B.Tech major/minor project — designed to be **explainable in a viva**, not just impressive on the surface.

---

## 1. Features

- Resume upload (PDF / DOCX / TXT) and Job Description input, with automatic extraction of skills, education, experience, projects, and JD requirements.
- Resume ↔ JD match score (`60% semantic similarity + 40% skill coverage`), clearly labeled as an app-generated estimate, not an ATS score.
- Configurable interview: type (Technical/HR/Behavioral/Mixed), difficulty, duration, number of questions, and focus areas.
- **Adaptive interview engine**: each question is generated one at a time based on the candidate's previous answers, scores, and weak/strong topics — not a pre-generated static list.
- **Knowledge-grounded evaluation (RAG)**: candidate answers are evaluated using context retrieved from a local knowledge base via Sentence-Transformers + FAISS, then scored by Gemini.
- Six-dimension answer scoring: technical correctness, completeness, relevance, clarity, communication, confidence indicator.
- Deterministic (non-LLM) communication analytics: word count, sentence count, filler words, vague phrases, answer length category.
- Final evaluation dashboard with radar chart, bar chart, and score cards.
- Skill-gap analysis: Strong / Needs Improvement / Missing, each with a reason, evidence, recommended topic, and a practice question.
- Personalized 7-day learning plan derived directly from the candidate's actual gaps.
- Interview history stored in SQLite, with trend charts across sessions.
- Downloadable PDF reports (Interview Report and Skill Gap Report).
- **Demo Mode**: run the entire application offline with deterministic sample data — no Gemini API key required. Ideal for live demonstrations.
- Modern top-navigation SaaS-style UI (no sidebar navigation).

---

## 2. Architecture

```
Resume + JD
    ↓
Document Processing        (document_loader.py — PDF/DOCX/TXT extraction)
    ↓
Resume/JD Analysis         (resume_analyzer.py, jd_analyzer.py — LLM + keyword fallback)
    ↓
Skill Extraction           (technical_skills, required_skills, preferred_skills)
    ↓
Interview Configuration    (type, difficulty, duration, question count, focus areas)
    ↓
Adaptive Interview Engine  (adaptive_interviewer.py — generates ONE question at a time)
    ↓
Candidate Answer           (text input; optional audio upload, gracefully degraded)
    ↓
RAG Knowledge Retrieval    (rag.py — Sentence-Transformers embeddings + FAISS search)
    ↓
LLM Evaluation             (evaluator.py — Gemini scores the answer using retrieved context)
    ↓
Performance Metrics        (analytics.py — aggregates scores + measured text stats)
    ↓
Skill Gap Analysis         (skill_gap.py — Resume vs JD vs Demonstrated skills)
    ↓
Personalized Learning Plan (7-day plan generated from actual gaps)
    ↓
Interview History + Dashboard (database.py — SQLite persistence, Plotly charts)
```

---

## 3. Technology Stack

| Layer | Technology | Why |
|---|---|---|
| UI | Streamlit + custom CSS | Rapid, Python-native UI; custom CSS gives a SaaS look without a separate frontend framework. |
| LLM | Google Gemini (`google-genai` SDK) | Free-tier friendly, strong instruction-following, native JSON mode for structured evaluation output. |
| Semantic similarity / RAG embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) | Small, fast, runs locally with no GPU, good semantic quality for short technical text. |
| Vector search | FAISS (`IndexFlatIP`) | Industry-standard, in-memory, exact nearest-neighbor search — simple to explain and fast at this knowledge-base scale. |
| Document parsing | pypdf, python-docx | Standard, dependency-light libraries for PDF/DOCX text extraction. |
| Storage | SQLite | Zero-configuration relational database, perfect for a local single-user project. |
| Data/analysis | Pandas, NumPy, scikit-learn | Standard data manipulation and could be extended with ML-based scoring. |
| Visualization | Plotly | Interactive, professional-looking charts (radar, bar, line) that render natively inside Streamlit. |
| Reports | ReportLab | Generates PDF reports in-memory without external binaries. |

---

## 4. Folder Structure

```
InterviewIQ/
│
├── app.py                     # Main Streamlit app (top navigation + all pages)
├── requirements.txt
├── .env.example
├── README.md
├── .gitignore
│
├── data/
│   └── interviewiq.db         # Created automatically on first run
│
├── knowledge_base/            # RAG source documents
│   ├── python.txt
│   ├── sql.txt
│   ├── machine_learning.txt
│   ├── data_structures.txt
│   ├── data_engineering.txt
│   ├── statistics.txt
│   ├── genai.txt
│   └── cloud.txt
│
├── src/
│   ├── __init__.py
│   ├── database.py            # SQLite persistence
│   ├── document_loader.py     # PDF/DOCX/TXT text extraction
│   ├── resume_analyzer.py     # Resume -> structured JSON (LLM + fallback)
│   ├── jd_analyzer.py         # JD -> structured JSON + Resume/JD match score
│   ├── embeddings.py          # Sentence-Transformers wrapper
│   ├── rag.py                 # FAISS-based retrieval pipeline
│   ├── llm.py                 # Gemini client wrapper (text + robust JSON)
│   ├── adaptive_interviewer.py# Adaptive next-question engine
│   ├── evaluator.py           # RAG-grounded answer evaluation
│   ├── skill_gap.py           # Skill classification + learning plan
│   ├── analytics.py           # Deterministic communication metrics + score aggregation
│   ├── report_generator.py    # PDF report generation
│   ├── ui.py                  # Custom CSS + top navbar + reusable widgets
│   └── sample_data.py         # Demo Mode sample resume/JD/question bank
│
└── assets/
```

---

## 5. Installation

```bash
# 1. Create and activate a virtual environment (Windows / VS Code)
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
copy .env.example .env
# then open .env and paste your Gemini API key
```

On macOS/Linux, use `source .venv/bin/activate` instead of `.venv\Scripts\activate`.

---

## 6. Environment Variables

Create a `.env` file (copied from `.env.example`) in the project root:

```
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.8-flash
```

- `GEMINI_API_KEY`: get one from https://aistudio.google.com/apikey
- `GEMINI_MODEL`: any current Gemini model name your key has access to. The app never hard-codes an obsolete model — it always reads this variable.

If `GEMINI_API_KEY` is missing, the app still runs fully in **Demo Mode**.

---

## 7. How to Run

```bash
streamlit run app.py
```

The app opens in your browser (typically at `http://localhost:8501`). Use the top navigation bar to move between Dashboard, Resume & JD, Interview Setup, Interview, Evaluation, Skill Gap, and History.

---

## 8. Demo Mode

Demo Mode is **ON by default** and toggled from the "⚙️ Settings" expander at the top of every page (deliberately not hidden in a sidebar). When ON:

- Sample resume and JD text (a Data Scientist / Data Engineer candidate) are used instead of file uploads.
- Interview questions come from a deterministic, hand-curated question bank instead of live Gemini calls — but the adaptive logic (which topic/difficulty to pick next) still runs for real.
- Answer evaluation uses a transparent rule-based scorer (keyword/length-based) instead of an LLM call.
- No RAG retrieval or API calls are made, so the entire app is demonstrable with **zero internet dependency and zero API cost**.

Turn Demo Mode OFF to use your real resume, a real JD, and live Gemini-powered questions/evaluation (requires `GEMINI_API_KEY`).

---

## 9. Gemini API Setup

1. Visit https://aistudio.google.com/apikey and create a free API key.
2. Paste it into `.env` as `GEMINI_API_KEY`.
3. Set `GEMINI_MODEL` to a model your key can access (e.g. `gemini-3.8-flash`).
4. Turn Demo Mode OFF in the app's Settings expander.

If the API key is invalid, quota is exceeded, or the network fails, every LLM-calling module (`resume_analyzer`, `jd_analyzer`, `adaptive_interviewer`, `evaluator`, `skill_gap`) automatically falls back to its deterministic logic instead of crashing, and a message explains what happened.

---

## 10. RAG (Retrieval-Augmented Generation) Explanation

1. Each `.txt` file in `knowledge_base/` is split into ~600-character overlapping chunks (`rag.py::_chunk_text`).
2. Each chunk is embedded into a 384-dimensional vector using `all-MiniLM-L6-v2` (Sentence-Transformers).
3. All chunk vectors are L2-normalized and stored in a FAISS `IndexFlatIP` index (inner product on normalized vectors = cosine similarity).
4. When an answer needs evaluation, the question + topic is embedded the same way, and the top-3 most similar chunks are retrieved.
5. Those chunks are injected into the Gemini evaluation prompt as **reference material**, and the LLM is instructed to check the candidate's answer against it.
6. The UI explicitly labels whether an evaluation was "knowledge-grounded" or not, and the app **never claims RAG guarantees factual correctness** — it only reduces the chance of the LLM inventing an incorrect standard when scoring.

---

## 11. Adaptive Interview Explanation

Instead of generating all questions upfront, `adaptive_interviewer.py` generates one question per turn using:

- The interview type, target difficulty, and focus areas chosen at setup.
- A running `skill_state` dict: `{topic: {avg_score, attempts}}`, updated after every answer.
- The full history of previous questions, answers, and scores (summarized into the prompt).

Adaptation rules (used both to steer the Gemini prompt, and as the deterministic Demo Mode fallback):
- Average score ≥ 75 on a topic → next question on that topic gets **harder**.
- Average score < 45 on a topic → next question is **easier** or a follow-up probing the same concept.
- Otherwise → continue at the current difficulty, picking the least-tested relevant topic.

This is a lightweight, explainable state machine — not a full reinforcement-learning policy — which makes it easy to describe in a viva while still producing genuinely adaptive behavior.

---

## 12. Evaluation Methodology

Each answer is scored 0–100 on six dimensions: technical correctness, completeness, relevance, clarity, communication, and a text-based confidence indicator. The LLM must return structured JSON (`llm.py::generate_json`), which is parsed defensively — a malformed or missing response never crashes the app; it falls back to `evaluator.py::_rule_based_fallback_score`, a transparent heuristic based on answer length and expected-keyword coverage.

Communication analytics (word count, sentence count, filler words, vague phrases) are computed **directly from the text**, not inferred by the LLM — the app is explicit in the UI about which numbers are measured vs. which are LLM judgments, and never claims to measure vocal tone, facial expression, or body language from text alone.

Session-level scores are aggregated in `analytics.py::aggregate_session_scores` by averaging each dimension across all answered questions; "Problem Solving" is an explainable proxy (`average of completeness, relevance, technical correctness`) rather than a separate opaque model.

---

## 13. Limitations

- Answer evaluation quality depends on Gemini's judgment; scores are for practice/self-assessment only, not an official hiring signal.
- The Resume-JD match score is a heuristic (semantic similarity + skill-keyword coverage), not an ATS replacement.
- Audio input is accepted in the UI but not transcribed locally; without a connected speech-to-text service, the app explicitly falls back to text-only evaluation rather than faking audio analysis.
- The RAG knowledge base is intentionally small (8 topic files) for a class project; retrieval quality and evaluation grounding are bounded by that scope.
- FAISS `IndexFlatIP` performs exact search, which would need to be swapped for an approximate index (e.g., IVF/HNSW) at much larger scale.
- Confidence indicator is inferred from answer text/tone only — it is not a measurement of the candidate's actual psychological state.

## 14. Future Improvements

- Real speech-to-text integration (e.g., via a cloud STT API) for genuine multimodal voice interviews.
- Video-based non-verbal analysis (with clear consent and disclosure).
- Fine-tuned or larger local knowledge base with document upload for custom RAG grounding.
- User accounts and multi-candidate support on top of the current single-user SQLite store.
- A/B testing different adaptive-difficulty policies and measuring interview outcome correlation.

---

## 15. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Ensure the virtual environment is activated and `pip install -r requirements.txt` completed successfully. |
| Gemini errors / quota exceeded | Turn Demo Mode ON in Settings, or check your API key/billing at https://aistudio.google.com. |
| FAISS/Sentence-Transformers fail to load | The app will still run; the Settings panel will show a warning, and evaluation will proceed without knowledge grounding. |
| PDF resume shows "no extractable text" | The PDF is likely a scanned image; try a DOCX/TXT version instead. |
| Port already in use | Run `streamlit run app.py --server.port 8502` to use a different port. |

---

**Disclaimer:** InterviewIQ is an educational project. Scores, match percentages, and skill-gap assessments are AI-assisted estimates for practice purposes and do not represent an official evaluation by any employer, university, or certification body.
