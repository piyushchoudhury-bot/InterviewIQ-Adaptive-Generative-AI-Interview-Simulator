"""
sample_data.py
--------------
Deterministic sample data used by Demo Mode and as a fallback whenever the
Gemini API is unavailable. Modeled on a Data Scientist / Data Engineer
candidate so the app can be demonstrated end-to-end offline.
"""

SAMPLE_RESUME_TEXT = """
ANANYA SHARMA
Aspiring Data Scientist / Data Engineer
Email: ananya.sharma@example.com | Phone: +91-9876543210

EDUCATION
B.Tech in Computer Science, ABC Institute of Technology, 2022-2026 (CGPA: 8.7/10)

TECHNICAL SKILLS
Programming: Python, SQL, Java (basic)
Data & ML: Pandas, NumPy, Scikit-learn, Matplotlib, Machine Learning, Statistics
BI & Visualization: Power BI, Plotly
Data Engineering: Basic ETL pipelines, MySQL, MongoDB
Cloud: AWS (S3, basic EC2 exposure)
GenAI: Prompt engineering, basic RAG concepts with LangChain
Tools: Git, GitHub, Jupyter Notebook, VS Code

PROJECTS
1. Customer Churn Prediction System
   Built a machine learning pipeline using Scikit-learn (Random Forest, Logistic
   Regression) to predict telecom customer churn with 84% accuracy. Performed
   feature engineering, handled class imbalance with SMOTE, and visualized
   results with Plotly dashboards.

2. Sales Data ETL Pipeline
   Designed a Python-based ETL pipeline to extract data from CSV/MySQL sources,
   clean and transform it with Pandas, and load it into a MySQL data warehouse
   modeled as a star schema. Automated the pipeline using Python scripts and
   Windows Task Scheduler.

3. Resume Screening Assistant (GenAI)
   Built a prototype using the Gemini API and a simple RAG pipeline with
   Sentence-Transformers and FAISS to match resumes against job descriptions
   and summarize candidate fit.

EXPERIENCE
Data Analyst Intern, XYZ Analytics Pvt Ltd (May 2025 - Jul 2025)
- Cleaned and analyzed sales datasets using Pandas and SQL, improving report
  turnaround time by 30%.
- Built interactive Power BI dashboards for the regional sales team.
- Collaborated with a team of 4 to migrate ad-hoc Excel reports into a
  centralized SQL-based reporting system.

CERTIFICATIONS
- Google Data Analytics Professional Certificate
- SQL for Data Science (Coursera)

SOFT SKILLS
Communication, teamwork, problem-solving, adaptability, time management
""".strip()

SAMPLE_JD_TEXT = """
Job Title: Junior Data Scientist / Data Engineer

We are looking for a motivated Junior Data Scientist / Data Engineer to join
our analytics team.

Responsibilities:
- Build and maintain ETL/ELT pipelines to process large datasets.
- Develop machine learning models for prediction and classification tasks.
- Write efficient SQL queries against our data warehouse.
- Collaborate with stakeholders to translate business problems into data
  solutions.
- Build dashboards and reports to communicate insights.
- Explore applications of Generative AI (RAG, LLMs) for internal tooling.

Required Skills:
- Strong proficiency in Python and SQL
- Experience with Pandas, NumPy, and Scikit-learn
- Solid understanding of statistics and machine learning fundamentals
- Experience building data pipelines (ETL/ELT)
- Familiarity with version control (Git)

Preferred Skills:
- Experience with cloud platforms (AWS, GCP, or Azure)
- Exposure to GenAI / LLM-based applications and RAG pipelines
- Experience with Power BI or similar visualization tools
- Knowledge of data warehousing concepts (star schema, dimensional modeling)
- Experience with big data tools such as Spark

Qualifications:
- Bachelor's degree in Computer Science, Data Science, or a related field
- 0-2 years of relevant experience (internships count)

Tools/Technologies: Python, SQL, Pandas, Scikit-learn, Power BI, AWS, Git
""".strip()

SAMPLE_RESUME_DATA = {
    "name": "Ananya Sharma",
    "education": ["B.Tech in Computer Science, ABC Institute of Technology, 2022-2026 (CGPA: 8.7/10)"],
    "experience": [
        "Data Analyst Intern, XYZ Analytics Pvt Ltd (May 2025 - Jul 2025)",
    ],
    "projects": [
        "Customer Churn Prediction System",
        "Sales Data ETL Pipeline",
        "Resume Screening Assistant (GenAI)",
    ],
    "technical_skills": [
        "Python", "SQL", "Java", "Pandas", "NumPy", "Scikit-learn", "Matplotlib",
        "Machine Learning", "Statistics", "Power BI", "Plotly", "MySQL", "MongoDB",
        "AWS", "GenAI", "RAG", "LangChain", "Git",
    ],
    "soft_skills": ["Communication", "Teamwork", "Problem-solving", "Adaptability", "Time management"],
    "certifications": [
        "Google Data Analytics Professional Certificate",
        "SQL for Data Science (Coursera)",
    ],
    "tools": ["Git", "GitHub", "Jupyter Notebook", "VS Code", "Power BI"],
}

SAMPLE_JD_DATA = {
    "required_skills": ["Python", "SQL", "Pandas", "NumPy", "Scikit-learn", "Statistics", "Machine Learning", "ETL", "Git"],
    "preferred_skills": ["AWS", "GenAI", "RAG", "Power BI", "Data Warehousing", "Spark", "Cloud"],
    "responsibilities": [
        "Build and maintain ETL/ELT pipelines",
        "Develop machine learning models",
        "Write efficient SQL queries",
        "Build dashboards and reports",
        "Explore GenAI applications",
    ],
    "qualifications": ["Bachelor's degree in Computer Science, Data Science, or related field", "0-2 years experience"],
    "tools": ["Python", "SQL", "Pandas", "Scikit-learn", "Power BI", "AWS", "Git"],
    "experience_requirement": "0-2 years (internships count)",
}

# Deterministic question bank used by the adaptive interviewer when the LLM
# and Demo Mode fallback is needed. Organized by skill and difficulty so the
# adaptive engine can still "adapt" deterministically in demo mode.
SAMPLE_QUESTION_BANK = {
    "Python": {
        "easy": {
            "question": "What is the difference between a list and a tuple in Python?",
            "expected_concepts": ["mutability", "syntax", "use cases"],
        },
        "medium": {
            "question": "Explain how Python's garbage collection works and what circular references are.",
            "expected_concepts": ["reference counting", "generational GC", "circular references"],
        },
        "hard": {
            "question": "How does the Global Interpreter Lock (GIL) affect multi-threaded Python programs, and how would you achieve true parallelism?",
            "expected_concepts": ["GIL", "multiprocessing", "I/O-bound vs CPU-bound"],
        },
    },
    "SQL": {
        "easy": {
            "question": "What is the difference between INNER JOIN and LEFT JOIN?",
            "expected_concepts": ["join types", "matching rows", "NULLs"],
        },
        "medium": {
            "question": "How would you find the second-highest salary in an Employees table using SQL?",
            "expected_concepts": ["subqueries", "window functions", "LIMIT/OFFSET"],
        },
        "hard": {
            "question": "Explain how you would optimize a slow-running query that joins three large tables.",
            "expected_concepts": ["indexing", "query plan", "EXPLAIN", "join order"],
        },
    },
    "Machine Learning": {
        "easy": {
            "question": "What is overfitting, and how can you detect it?",
            "expected_concepts": ["train/test gap", "high variance", "cross-validation"],
        },
        "medium": {
            "question": "Explain precision and recall, and describe a scenario where you would prioritize one over the other.",
            "expected_concepts": ["precision", "recall", "class imbalance", "business context"],
        },
        "hard": {
            "question": "Walk through how gradient boosting works and how it differs from random forests.",
            "expected_concepts": ["sequential ensembling", "residuals", "bagging vs boosting"],
        },
    },
    "Data Structures": {
        "easy": {
            "question": "What is the time complexity of accessing an element in an array vs. a linked list?",
            "expected_concepts": ["O(1) array access", "O(n) linked list traversal"],
        },
        "medium": {
            "question": "How would you detect a cycle in a linked list?",
            "expected_concepts": ["Floyd's algorithm", "slow/fast pointers"],
        },
        "hard": {
            "question": "Explain how a hash table handles collisions and what factors affect its performance.",
            "expected_concepts": ["chaining", "open addressing", "load factor"],
        },
    },
    "Data Engineering": {
        "easy": {
            "question": "What is the difference between ETL and ELT?",
            "expected_concepts": ["extract/transform/load order", "modern warehouses"],
        },
        "medium": {
            "question": "Describe a star schema and why it's used in data warehousing.",
            "expected_concepts": ["fact table", "dimension tables", "query performance"],
        },
        "hard": {
            "question": "How would you design a pipeline to handle late-arriving and duplicate data at scale?",
            "expected_concepts": ["idempotency", "deduplication", "watermarking"],
        },
    },
    "Cloud": {
        "easy": {
            "question": "What is the difference between IaaS, PaaS, and SaaS?",
            "expected_concepts": ["service models", "level of abstraction"],
        },
        "medium": {
            "question": "When would you choose serverless functions over containers?",
            "expected_concepts": ["event-driven workloads", "cold starts", "operational overhead"],
        },
        "hard": {
            "question": "How would you design a highly available, multi-region architecture on the cloud?",
            "expected_concepts": ["redundancy", "load balancing", "failover", "data replication"],
        },
    },
    "GenAI": {
        "easy": {
            "question": "What is Retrieval-Augmented Generation (RAG), and why is it used?",
            "expected_concepts": ["retrieval", "grounding", "reducing hallucination"],
        },
        "medium": {
            "question": "Explain how embeddings are used in a semantic search system.",
            "expected_concepts": ["vector representation", "cosine similarity", "ANN search"],
        },
        "hard": {
            "question": "What are the tradeoffs between fine-tuning, prompting, and RAG for adapting an LLM to a domain?",
            "expected_concepts": ["cost", "flexibility", "knowledge freshness"],
        },
    },
    "Statistics": {
        "easy": {
            "question": "What is a p-value, and how would you explain it to a non-technical stakeholder?",
            "expected_concepts": ["hypothesis testing", "significance", "common misinterpretation"],
        },
        "medium": {
            "question": "How would you design an A/B test to evaluate a new website feature?",
            "expected_concepts": ["sample size", "randomization", "statistical power"],
        },
        "hard": {
            "question": "Explain the bias-variance tradeoff and how it applies to model selection.",
            "expected_concepts": ["bias", "variance", "regularization", "generalization"],
        },
    },
}

SAMPLE_HR_QUESTIONS = {
    "easy": {"question": "Tell me about yourself and why you're interested in this role.", "expected_concepts": ["clarity", "relevance", "motivation"]},
    "medium": {"question": "Describe a time you faced a conflict while working in a team. How did you resolve it?", "expected_concepts": ["STAR method", "conflict resolution", "self-awareness"]},
    "hard": {"question": "Tell me about a time you failed at something important. What did you learn from it?", "expected_concepts": ["accountability", "growth mindset", "specific example"]},
}
