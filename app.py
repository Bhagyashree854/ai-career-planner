# ==========================================
# IMPORTS
# ==========================================
import streamlit as st
import re
import random
import nltk
import PyPDF2
import plotly.express as px
import pandas as pd

from fpdf import FPDF
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# NLTK DOWNLOADS
# ==========================================
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="AI Career Planner",
    layout="wide",
    page_icon="🚀"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>

.stApp{
    background: linear-gradient(to right,#f8fafc,#eef2ff);
}

h1,h2,h3,h4{
    color:#0f172a !important;
}

div[data-testid="metric-container"]{
    background:white;
    border-radius:15px;
    padding:15px;
    border:1px solid #dbeafe;
    box-shadow:0px 4px 10px rgba(0,0,0,0.05);
}

.stButton > button{
    background:linear-gradient(to right,#2563eb,#7c3aed);
    color:white;
    border:none;
    border-radius:10px;
    padding:10px 18px;
    font-weight:bold;
}

.quiz-card{
    background:white;
    padding:18px;
    border-radius:12px;
    margin-bottom:15px;
    border:1px solid #dbeafe;
}

.answer-box{
    background:#dcfce7;
    padding:12px;
    border-radius:10px;
    margin-top:10px;
}

.roadmap-card{
    background:white;
    padding:20px;
    border-radius:15px;
    margin-bottom:15px;
    border:1px solid #dbeafe;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# DATA
# ==========================================
career_categories = {

    "AI & Data": [
        "data_analyst",
        "data_scientist",
        "machine_learning_engineer"
    ],

    "Software": [
        "frontend_developer",
        "backend_developer",
        "full_stack_developer"
    ],

    "Cloud": [
        "cloud_engineer",
        "devops_engineer"
    ]
}

career_skill_map = {

    "data_analyst": [
        "python",
        "sql",
        "excel",
        "statistics",
        "pandas",
        "power bi",
        "tableau"
    ],

    "data_scientist": [
        "python",
        "numpy",
        "pandas",
        "machine learning",
        "statistics",
        "sql",
        "deep learning"
    ],

    "machine_learning_engineer": [
        "python",
        "machine learning",
        "deep learning",
        "tensorflow",
        "pytorch"
    ],

    "frontend_developer": [
        "html",
        "css",
        "javascript",
        "react"
    ],

    "backend_developer": [
        "python",
        "nodejs",
        "sql",
        "api",
        "database"
    ],

    "full_stack_developer": [
        "html",
        "css",
        "javascript",
        "react",
        "nodejs",
        "database"
    ],

    "cloud_engineer": [
        "aws",
        "docker",
        "kubernetes",
        "linux"
    ],

    "devops_engineer": [
        "docker",
        "kubernetes",
        "ci/cd",
        "aws"
    ]
}

# ==========================================
# PROJECTS
# ==========================================
project_recommendations = {

    "python": [
        "Student Management System",
        "Expense Tracker",
        "Weather App"
    ],

    "sql": [
        "Library Database System",
        "Hospital Management System",
        "Sales Analytics Database"
    ],

    "machine learning": [
        "House Price Prediction",
        "Spam Email Classifier",
        "Recommendation System"
    ],

    "html": [
        "Portfolio Website",
        "Landing Page",
        "Restaurant Website"
    ],

    "css": [
        "Responsive Dashboard",
        "Animated Website",
        "Modern Portfolio"
    ],

    "javascript": [
        "Quiz Application",
        "Weather App",
        "ToDo App"
    ]
}

# ==========================================
# QUIZ DATA
# ==========================================
quiz_data = {

    "python": [

        {
            "question": "Which keyword defines function?",
            "options": ["func", "define", "def", "lambda"],
            "answer": "def"
        },

        {
            "question": "Which datatype is immutable?",
            "options": ["list", "tuple", "dict", "set"],
            "answer": "tuple"
        },

        {
            "question": "Which function prints output?",
            "options": ["show()", "print()", "echo()", "display()"],
            "answer": "print()"
        }
    ]
}

# ==========================================
# DEPENDENCIES
# ==========================================
dependencies = {

    "machine learning": ["python"],
    "deep learning": ["machine learning"]
}

# ==========================================
# STOPWORDS
# ==========================================
stop_words = set(stopwords.words("english"))

# ==========================================
# SESSION STATE
# ==========================================
defaults = {

    "analysis_done": False,
    "matched": [],
    "missing": [],
    "readiness": 0,
    "resume_score": 0,
    "user_skills": "",
    "chat_history": [],
    "selected_mcqs": []
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value

# ==========================================
# PREPROCESS
# ==========================================
def preprocess(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    tokens = word_tokenize(text)

    clean = [

        w for w in tokens
        if w not in stop_words
    ]

    return " ".join(clean)

# ==========================================
# PDF EXTRACTION
# ==========================================
def extract_pdf(pdf):

    reader = PyPDF2.PdfReader(pdf)

    text = ""

    for page in reader.pages:

        content = page.extract_text()

        if content:
            text += content

    return text

# ==========================================
# ATS FEEDBACK
# ==========================================
def ats_feedback(resume_text):

    suggestions = []

    text = resume_text.lower()

    if "skills" not in text:
        suggestions.append("Add Skills Section")

    if "projects" not in text:
        suggestions.append("Add Projects Section")

    if "experience" not in text:
        suggestions.append("Add Experience Section")

    return suggestions

# ==========================================
# EXTRACT SKILLS
# ==========================================
def extract_skills(text):

    text = preprocess(text)

    found = set()

    for skills in career_skill_map.values():

        for skill in skills:

            if all(word in text for word in skill.split()):
                found.add(skill)

    return list(found)

# ==========================================
# ANALYSIS
# ==========================================
def analyze(skills, role):

    required = career_skill_map[role]

    text = preprocess(skills)

    matched = [
        s for s in required
        if s in text
    ]

    missing = [
        s for s in required
        if s not in text
    ]

    readiness = round(
        len(matched) / len(required) * 100,
        2
    )

    return matched, missing, readiness

# ==========================================
# SIMILARITY
# ==========================================
def similarity(user, role):

    required = " ".join(career_skill_map[role])

    tfidf = TfidfVectorizer().fit_transform(
        [user, required]
    )

    score = cosine_similarity(
        tfidf[0:1],
        tfidf[1:2]
    )[0][0]

    return round(score * 100, 2)

# ==========================================
# ROLE RECOMMENDATION
# ==========================================
def recommend_roles(user_skills):

    recommendations = []

    for role, skills in career_skill_map.items():

        required_text = " ".join(skills)

        tfidf = TfidfVectorizer().fit_transform(
            [user_skills, required_text]
        )

        score = cosine_similarity(
            tfidf[0:1],
            tfidf[1:2]
        )[0][0]

        recommendations.append(
            (role, round(score * 100, 2))
        )

    recommendations.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return recommendations[:3]

# ==========================================
# ROADMAP SORTING
# ==========================================
def sort_skills(skills):

    visited = set()

    order = []

    def dfs(skill):

        if skill in visited:
            return

        visited.add(skill)

        for dep in dependencies.get(skill, []):

            if dep in skills:
                dfs(dep)

        order.append(skill)

    for skill in skills:
        dfs(skill)

    return order

# ==========================================
# RESOURCES
# ==========================================
def resources(skill):

    q = skill.replace(" ", "+")

    return {

        "course":
        f"https://www.youtube.com/results?search_query={q}+course",

        "docs":
        f"https://www.google.com/search?q={q}+documentation",

        "practice":
        f"https://www.google.com/search?q={q}+projects"
    }

# ==========================================
# GENERATE PDF
# ==========================================
def generate_pdf(role, matched, missing, readiness):

    pdf = FPDF()

    pdf.add_page()

    pdf.set_font("Arial", "B", 20)

    pdf.cell(
        200,
        10,
        txt="AI Career Planner Report",
        ln=True,
        align="C"
    )

    pdf.ln(10)

    pdf.set_font("Arial", size=14)

    pdf.cell(
        200,
        10,
        txt=f"Target Role: {role.replace('_',' ').title()}",
        ln=True
    )

    pdf.cell(
        200,
        10,
        txt=f"Career Readiness: {readiness}%",
        ln=True
    )

    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)

    pdf.cell(
        200,
        10,
        txt="Matched Skills:",
        ln=True
    )

    pdf.set_font("Arial", size=12)

    for skill in matched:

        pdf.cell(
            200,
            8,
            txt=f"- {skill}",
            ln=True
        )

    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)

    pdf.cell(
        200,
        10,
        txt="Missing Skills:",
        ln=True
    )

    pdf.set_font("Arial", size=12)

    for skill in missing:

        pdf.cell(
            200,
            8,
            txt=f"- {skill}",
            ln=True
        )

    filename = "career_report.pdf"

    pdf.output(filename)

    return filename

# ==========================================
# UI
# ==========================================
st.title("🚀 AI Career Planner")

category = st.selectbox(
    "Select Category",
    list(career_categories.keys())
)

role = st.selectbox(
    "Select Target Role",
    career_categories[category]
)

pdf_file = st.file_uploader(
    "Upload Resume",
    type=["pdf"]
)

skills_text = ""

if pdf_file:

    text = extract_pdf(pdf_file)

    auto_skills = extract_skills(text)

    st.success("Resume Processed Successfully")

    st.write("Extracted Skills:", auto_skills)

    skills_text = ", ".join(auto_skills)

skills_input = st.text_area(
    "Your Skills",
    value=skills_text
)

# ==========================================
# ANALYZE BUTTON
# ==========================================
if st.button("Analyze"):

    st.session_state.user_skills = skills_input

    matched, missing, readiness = analyze(
        skills_input,
        role
    )

    st.session_state.matched = matched
    st.session_state.missing = missing
    st.session_state.readiness = readiness
    st.session_state.resume_score = readiness
    st.session_state.analysis_done = True

# ==========================================
# RESULTS
# ==========================================
if st.session_state.analysis_done:

    sim = similarity(
        st.session_state.user_skills,
        role
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Readiness",
        f"{st.session_state.readiness}%"
    )

    col2.metric(
        "Similarity",
        f"{sim}%"
    )

    col3.metric(
        "Resume Score",
        f"{st.session_state.resume_score}%"
    )

    # ======================================
    # RECOMMENDED ROLES
    # ======================================
    st.subheader("🎯 Recommended Roles")

    recommendations = recommend_roles(
        st.session_state.user_skills
    )

    for idx, (r, score) in enumerate(recommendations, start=1):

        st.markdown(
            f"### {idx}. {r.replace('_',' ').title()}"
        )

        st.progress(int(score))

        st.write(f"Match Score: {score}%")

    # ======================================
    # MATCHED SKILLS
    # ======================================
    with st.expander("✅ Matched Skills"):

        for skill in st.session_state.matched:
            st.success(skill)

    # ======================================
    # MISSING SKILLS
    # ======================================
    with st.expander("❌ Missing Skills"):

        for skill in st.session_state.missing:
            st.error(skill)

    # ======================================
    # ROADMAP
    # ======================================
    st.subheader("📚 Adaptive Learning Roadmap")

    ordered = sort_skills(
        st.session_state.missing
    )

    for i, skill in enumerate(ordered):

        st.markdown(f"""
<div class="roadmap-card">

<h3>Week {i*2+1}-{i*2+2}: {skill.title()}</h3>

</div>
""", unsafe_allow_html=True)

        res = resources(skill)

        st.markdown(f"- [Course]({res['course']})")
        st.markdown(f"- [Documentation]({res['docs']})")
        st.markdown(f"- [Practice Projects]({res['practice']})")

        if skill in project_recommendations:

            st.write("### Recommended Projects")

            for project in project_recommendations[skill]:

                st.write(f"• {project}")

    # ======================================
    # CHART
    # ======================================
    df = pd.DataFrame({

        "Type": ["Matched", "Missing"],

        "Count": [
            len(st.session_state.matched),
            len(st.session_state.missing)
        ]
    })

    fig = px.bar(
        df,
        x="Type",
        y="Count",
        title="Skill Analysis"
    )

    st.plotly_chart(fig)

    # ======================================
    # ATS FEEDBACK
    # ======================================
    st.subheader("📄 ATS Resume Analysis")

    if pdf_file:

        ats = ats_feedback(text)

        if ats:

            for tip in ats:
                st.warning(tip)

        else:
            st.success("Your Resume Looks ATS Friendly")

# ==========================================
# MCQ SECTION
# ==========================================
st.subheader("🧠 Skill Testing MCQs")

mcq_count = st.slider(
    "Select Number of MCQs",
    1,
    5,
    3
)

if st.button("Generate MCQs"):

    questions_pool = []

    for skill in st.session_state.matched:

        if skill in quiz_data:
            questions_pool.extend(
                quiz_data[skill]
            )

    random.shuffle(questions_pool)

    st.session_state.selected_mcqs = questions_pool[:mcq_count]

# ==========================================
# DISPLAY MCQS
# ==========================================
if len(st.session_state.selected_mcqs) > 0:

    user_answers = []

    for idx, q in enumerate(st.session_state.selected_mcqs):

        st.markdown(
            '<div class="quiz-card">',
            unsafe_allow_html=True
        )

        answer = st.radio(
            f"Q{idx+1}. {q['question']}",
            q["options"],
            key=f"q_{idx}"
        )

        user_answers.append({

            "question": q["question"],
            "selected": answer,
            "correct": q["answer"]
        })

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    if st.button("Submit MCQ Test"):

        score = 0

        for item in user_answers:

            if item["selected"] == item["correct"]:
                score += 1

        percent = round(
            (score / len(user_answers)) * 100,
            2
        )

        st.success(f"🎯 MCQ Score: {percent}%")

        st.subheader("✅ Correct Answers")

        for idx, item in enumerate(user_answers):

            st.markdown(f"""
<div class="answer-box">

<b>Q{idx+1}:</b> {item['question']}<br><br>

<b>Your Answer:</b> {item['selected']}<br>

<b>Correct Answer:</b> {item['correct']}

</div>
""", unsafe_allow_html=True)

# ==========================================
# CHATBOT
# ==========================================
st.subheader("🤖 AI Career Mentor")

for chat in st.session_state.chat_history:

    with st.chat_message(chat["role"]):

        st.markdown(chat["message"])

user_prompt = st.chat_input(
    "Ask your career question..."
)

if user_prompt:

    st.session_state.chat_history.append({

        "role": "user",
        "message": user_prompt
    })

    response = f"""
### 🚀 AI Career Guidance

You asked:
**{user_prompt}**

### 📌 Suggestions

- Build strong projects
- Practice coding regularly
- Improve communication skills
- Stay consistent with roadmap

### 🎯 Target Role
{role.replace('_',' ').title()}
"""

    st.session_state.chat_history.append({

        "role": "assistant",
        "message": response
    })

    st.rerun()

# ==========================================
# PDF DOWNLOAD
# ==========================================
st.markdown("---")

st.subheader("📥 Download Career Report")

if st.button("Generate PDF Report"):

    file = generate_pdf(
        role,
        st.session_state.matched,
        st.session_state.missing,
        st.session_state.readiness
    )

    with open(file, "rb") as f:

        st.download_button(
            label="⬇ Download AI Career Report",
            data=f,
            file_name="AI_Career_Report.pdf",
            mime="application/pdf"
        )