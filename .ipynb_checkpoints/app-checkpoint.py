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
import nltk

# Download punkt
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Download punkt_tab
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

# Download stopwords
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
# UI DESIGN
# ==========================================
st.markdown("""
<style>

.stApp{
    background: linear-gradient(to right,#f8fafc,#eef2ff);
}

h1,h2,h3,h4,h5,h6{
    color:#0f172a !important;
    font-weight:700;
}

p,label,div{
    color:#1e293b;
}

div[data-testid="metric-container"]{
    background:white;
    border-radius:16px;
    padding:18px;
    border:1px solid #dbeafe;
    box-shadow:0 4px 12px rgba(0,0,0,0.08);
}

.stButton > button{
    background:linear-gradient(to right,#2563eb,#7c3aed);
    color:white;
    border:none;
    border-radius:12px;
    padding:10px 18px;
    font-weight:bold;
}

.stTextArea textarea{
    background:white !important;
    color:black !important;
}

.quiz-card{
    background:white;
    padding:18px;
    border-radius:15px;
    border:1px solid #dbeafe;
    margin-bottom:15px;
}

.answer-box{
    background:#dcfce7;
    padding:12px;
    border-radius:10px;
    margin-top:10px;
    color:#166534;
    font-weight:bold;
}

.roadmap-card{
    background:white;
    padding:20px;
    border-radius:18px;
    margin-bottom:20px;
    border:1px solid #dbeafe;
    box-shadow:0 4px 10px rgba(0,0,0,0.05);
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

    "excel": [
        "Sales Dashboard in Excel",
        "Employee Salary Analysis",
        "Financial Report Automation",
        "Inventory Tracker",
        "Expense Management System"
    ],

    "statistics": [
        "Customer Survey Analysis",
        "Market Trend Analysis",
        "A/B Testing Project",
        "Business Data Analysis",
        "Risk Analysis Dashboard"
    ],

    "pandas": [
        "Netflix Data Analysis",
        "IPL Data Analysis",
        "COVID-19 Data Exploration",
        "Student Dataset Analysis",
        "Stock Market Analysis"
    ],

    "tableau": [
        "Interactive Data Dashboard",
        "Customer Segmentation Dashboard",
        "Retail Analytics Dashboard",
        "HR Analytics Dashboard",
        "Business KPI Dashboard"
    ],

    "python": [
        "Student Management System",
        "Expense Tracker",
        "Chat Application",
        "Automation Script",
        "Weather App"
    ],

    "sql": [
        "Library Database System",
        "Hospital Management Database",
        "Sales Analytics Database",
        "Employee Database",
        "Banking Database"
    ]
}

# ==========================================
# DEPENDENCIES
# ==========================================
dependencies = {

    "machine learning": [
        "python"
    ],

    "deep learning": [
        "machine learning"
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
            "question": "Which symbol is used for comments?",
            "options": ["#", "//", "@", "**"],
            "answer": "#"
        },

        {
            "question": "Which datatype is immutable?",
            "options": ["list", "tuple", "set", "dict"],
            "answer": "tuple"
        },

        {
            "question": "Which loop is used for iteration?",
            "options": ["iterate", "for", "repeat", "loop"],
            "answer": "for"
        },

        {
            "question": "Which library used for data analysis?",
            "options": ["numpy", "pandas", "pygame", "tkinter"],
            "answer": "pandas"
        },

        {
            "question": "Which operator is used for power?",
            "options": ["^", "**", "//", "%"],
            "answer": "**"
        },

        {
            "question": "Which datatype stores key-value pairs?",
            "options": ["list", "tuple", "dict", "set"],
            "answer": "dict"
        },

        {
            "question": "Which keyword is used for conditional statements?",
            "options": ["loop", "if", "switch", "case"],
            "answer": "if"
        },

        {
            "question": "Which function prints output?",
            "options": ["show()", "display()", "print()", "echo()"],
            "answer": "print()"
        },

        {
            "question": "Which keyword is used to import libraries?",
            "options": ["include", "import", "using", "require"],
            "answer": "import"
        }
    ]
}

# ==========================================
# STOPWORDS
# ==========================================
stop_words = set(stopwords.words('english'))

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
    "chat_history": []
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
        r'[^a-zA-Z\s]',
        '',
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

        t = page.extract_text()

        if t:
            text += t

    return text

# ==========================================
# ATS FEEDBACK
# ==========================================
def ats_feedback(resume_text):

    suggestions = []

    text = resume_text.lower()

    if "projects" not in text:
        suggestions.append("Add Projects Section")

    if "skills" not in text:
        suggestions.append("Add Skills Section")

    if "experience" not in text:
        suggestions.append("Add Experience Section")

    return suggestions

# ==========================================
# SKILL EXTRACTION
# ==========================================
def extract_skills(text):

    text = preprocess(text)

    found = set()

    for skills in career_skill_map.values():

        for skill in skills:

            if all(
                word in text
                for word in skill.split()
            ):
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

    required = " ".join(
        career_skill_map[role]
    )

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
            (
                role,
                round(score * 100, 2)
            )
        )

    recommendations.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return recommendations[:3]

# ==========================================
# ROADMAP SORT
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
# PDF REPORT
# ==========================================
def generate_pdf(role, matched, missing, readiness):

    pdf = FPDF()

    pdf.add_page()

    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="AI Career Report", ln=True)

    pdf.cell(200, 10, txt=f"Target Role: {role}", ln=True)

    pdf.cell(200, 10, txt=f"Readiness: {readiness}%", ln=True)

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

    # ==========================================
    # RECOMMENDED ROLES
    # ==========================================
    st.subheader("🎯 Recommended Roles")

    recommendations = recommend_roles(
        st.session_state.user_skills
    )

    for idx, (r, score) in enumerate(
        recommendations,
        start=1
    ):

        st.markdown(
            f"### {idx}. {r.replace('_',' ').title()}"
        )

        st.progress(int(score))

        st.write(f"Match Score: {score}%")

    # ==========================================
    # MATCHED SKILLS
    # ==========================================
    with st.expander("✅ Matched Skills"):

        for skill in st.session_state.matched:
            st.success(skill)

    # ==========================================
    # MISSING SKILLS
    # ==========================================
    with st.expander("❌ Missing Skills"):

        for skill in st.session_state.missing:
            st.error(skill)

    # ==========================================
    # ROADMAP
    # ==========================================
    st.subheader("📚 Adaptive Learning Roadmap")

    ordered = sort_skills(
        st.session_state.missing
    )

    for i, skill in enumerate(ordered):

        st.markdown(
            f"""
<div class="roadmap-card">

### Week {i*2+1}-{i*2+2}: {skill.title()}

</div>
""",
            unsafe_allow_html=True
        )

        res = resources(skill)

        st.markdown(f"- [Course]({res['course']})")

        st.markdown(f"- [Documentation]({res['docs']})")

        st.markdown(f"- [Practice Projects]({res['practice']})")

        st.write("### Recommended Projects")

        if skill in project_recommendations:

            projects = project_recommendations[skill]

            selected_projects = random.sample(
                projects,
                min(3, len(projects))
            )

            for project in selected_projects:
                st.write(f"• {project}")

        else:
            st.info("Projects will be added soon.")

        st.markdown("---")

    # ==========================================
    # ATS FEEDBACK
    # ==========================================
    st.subheader("📄 ATS Resume Analysis")

    if pdf_file:

        ats = ats_feedback(text)

        if ats:

            for tip in ats:
                st.warning(tip)

        else:
            st.success("Your Resume Looks ATS Friendly")

# ==========================================
# INITIALIZE MCQ SESSION
# ==========================================
if "selected_mcqs" not in st.session_state:
    st.session_state.selected_mcqs = []

if "show_answers" not in st.session_state:
    st.session_state.show_answers = False

# ==========================================
# MCQ SECTION
# ==========================================
st.subheader("🧠 Skill Testing MCQs")

mcq_count = st.slider(
    "Select Number of MCQs",
    5,
    15,
    5
)

# ------------------------------------------
# GENERATE QUESTIONS BUTTON
# ------------------------------------------
if st.button("Generate MCQs"):

    questions_pool = []

    for skill in st.session_state.matched:

        if skill in quiz_data:

            questions_pool.extend(
                quiz_data[skill]
            )

    random.shuffle(questions_pool)

    if mcq_count > len(questions_pool):
        mcq_count = len(questions_pool)

    st.session_state.selected_mcqs = questions_pool[:mcq_count]

    st.session_state.show_answers = False

# ------------------------------------------
# DISPLAY QUESTIONS
# ------------------------------------------
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
            key=f"question_{idx}"
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

    # --------------------------------------
    # SUBMIT TEST
    # --------------------------------------
    if st.button("Submit MCQ Test"):

        st.session_state.show_answers = True

        score = 0

        for item in user_answers:

            if item["selected"] == item["correct"]:
                score += 1

        percent = round(
            (score / len(user_answers)) * 100,
            2
        )

        st.success(
            f"🎯 MCQ Score: {percent}%"
        )

        if percent >= 80:
            st.balloons()

        st.subheader("✅ Correct Answers")

        for idx, item in enumerate(user_answers):

            st.markdown(f"""
<div class="answer-box">

<b>Q{idx+1}:</b> {item['question']}<br><br>

<b>Your Answer:</b> {item['selected']}<br>

<b>Correct Answer:</b> {item['correct']}

</div>
""", unsafe_allow_html=True)

else:

    st.info("Click 'Generate MCQs' to start test.")
    
    # ==========================================
    # CHART
    # ==========================================
    df = pd.DataFrame({

        "Type": [
            "Matched",
            "Missing"
        ],

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

# ==========================================
# CHATBOT SECTION
# ==========================================
st.subheader("🤖 AI Career Mentor")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------------------------------
# DISPLAY CHAT HISTORY
# ------------------------------------------
for chat in st.session_state.chat_history:

    with st.chat_message(chat["role"]):

        st.markdown(chat["message"])

# ------------------------------------------
# USER INPUT
# ------------------------------------------
user_prompt = st.chat_input(
    "Ask your career question..."
)

# ------------------------------------------
# CHATBOT RESPONSE
# ------------------------------------------
if user_prompt:

    st.session_state.chat_history.append({

        "role": "user",
        "message": user_prompt
    })

    response = f"""
### 🚀 AI Career Guidance

You asked:
**{user_prompt}**

### 📌 Recommended Advice

- Build strong projects
- Improve missing skills
- Practice interview questions
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
# PDF REPORT
# ==========================================
def generate_pdf(role, matched, missing, readiness):

    pdf = FPDF()

    pdf.add_page()

    # --------------------------------------
    # TITLE
    # --------------------------------------
    pdf.set_fill_color(37, 99, 235)

    pdf.set_text_color(255, 255, 255)

    pdf.set_font("Arial", "B", 22)

    pdf.cell(
        190,
        15,
        txt="AI Career Planner Report",
        ln=True,
        align="C",
        fill=True
    )

    pdf.ln(10)

    # --------------------------------------
    # RESET COLORS
    # --------------------------------------
    pdf.set_text_color(0, 0, 0)

    # --------------------------------------
    # ROLE SECTION
    # --------------------------------------
    pdf.set_font("Arial", "B", 16)

    pdf.cell(
        200,
        10,
        txt=f"Target Role: {role.replace('_', ' ').title()}",
        ln=True
    )

    pdf.ln(4)

    # --------------------------------------
    # READINESS SCORE
    # --------------------------------------
    pdf.set_font("Arial", "B", 14)

    pdf.cell(
        200,
        10,
        txt=f"Career Readiness Score: {readiness}%",
        ln=True
    )

    pdf.ln(6)

    # --------------------------------------
    # MATCHED SKILLS
    # --------------------------------------
    pdf.set_fill_color(220, 252, 231)

    pdf.set_font("Arial", "B", 15)

    pdf.cell(
        190,
        10,
        txt="Matched Skills",
        ln=True,
        fill=True
    )

    pdf.set_font("Arial", "", 13)

    pdf.ln(3)

    if len(matched) > 0:

        for skill in matched:

            pdf.cell(
                200,
                8,
                txt=f"- {skill.title()}",
                ln=True
            )

    else:

        pdf.cell(
            200,
            8,
            txt="No matched skills found.",
            ln=True
        )

    pdf.ln(6)

    # --------------------------------------
    # MISSING SKILLS
    # --------------------------------------
    pdf.set_fill_color(254, 226, 226)

    pdf.set_font("Arial", "B", 15)

    pdf.cell(
        190,
        10,
        txt="Missing Skills",
        ln=True,
        fill=True
    )

    pdf.set_font("Arial", "", 13)

    pdf.ln(3)

    if len(missing) > 0:

        for skill in missing:

            pdf.cell(
                200,
                8,
                txt=f"- {skill.title()}",
                ln=True
            )

    else:

        pdf.cell(
            200,
            8,
            txt="No missing skills.",
            ln=True
        )

    pdf.ln(8)

    # --------------------------------------
    # CAREER SUGGESTION
    # --------------------------------------
    pdf.set_fill_color(219, 234, 254)

    pdf.set_font("Arial", "B", 15)

    pdf.cell(
        190,
        10,
        txt="Career Suggestions",
        ln=True,
        fill=True
    )

    pdf.set_font("Arial", "", 12)

    suggestions = [

        "Build strong real-world projects.",
        "Improve communication skills.",
        "Practice interview preparation regularly.",
        "Complete roadmap topics consistently.",
        "Create ATS-friendly resume."
    ]

    pdf.ln(3)

    for tip in suggestions:

        pdf.multi_cell(
            0,
            8,
            txt=f"- {tip}"
        )

    pdf.ln(8)

    # --------------------------------------
    # FOOTER
    # --------------------------------------
    pdf.set_font("Arial", "I", 10)

    pdf.set_text_color(100, 100, 100)

    pdf.cell(
        200,
        10,
        txt="Generated by AI Career Planner",
        ln=True,
        align="C"
    )

    # --------------------------------------
    # SAVE FILE
    # --------------------------------------
    filename = "career_report.pdf"

    pdf.output(filename)

    return filename

# ==========================================
# PDF DOWNLOAD SECTION
# ==========================================
st.markdown("---")

st.subheader("📥 Download Career Report")

st.write(
    "Generate a professional AI Career Report with matched skills, missing skills, readiness score, and career suggestions."
)

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