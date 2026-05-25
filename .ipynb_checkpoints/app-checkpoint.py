import streamlit as st
import re
import nltk
import PyPDF2
import plotly.express as px
import pandas as pd

# ---------------------------
# DOWNLOADS
# ---------------------------
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

from fpdf import FPDF

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# DOWNLOADS
# ---------------------------
nltk.download('punkt')
nltk.download('stopwords')

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="AI Career Planner",
    layout="wide"
)

# ---------------------------
# DATA
# ---------------------------
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

# ---------------------------
# PROJECTS
# ---------------------------
project_recommendations = {

    "excel": [
        "Sales Dashboard in Excel",
        "Employee Salary Analysis",
        "Financial Report Automation"
    ],

    "statistics": [
        "Customer Survey Analysis",
        "Market Trend Analysis",
        "A/B Testing Project"
    ],

    "pandas": [
        "Netflix Data Analysis",
        "IPL Data Analysis",
        "COVID-19 Data Exploration"
    ],

    "power bi": [
        "Business Sales Dashboard",
        "HR Analytics Dashboard",
        "Finance KPI Dashboard"
    ],

    "tableau": [
        "Interactive Data Dashboard",
        "Customer Segmentation Dashboard",
        "Retail Analytics Dashboard"
    ],

    "sql": [
        "Library Database System",
        "Hospital Management Database",
        "Sales Analytics Database"
    ],

    "python": [
        "Student Management System",
        "Expense Tracker",
        "Chat Application"
    ],

    "numpy": [
        "Matrix Operations Project",
        "Numerical Computing System"
    ],

    "machine learning": [
        "House Price Prediction",
        "Spam Email Classifier",
        "Student Performance Predictor"
    ],

    "deep learning": [
        "Image Classification",
        "Face Mask Detection",
        "Emotion Detection System"
    ],

    "tensorflow": [
        "CNN Image Classifier",
        "AI Object Detection System",
        "Handwritten Digit Recognition"
    ],

    "pytorch": [
        "Custom Neural Network",
        "AI Image Generator",
        "Object Detection Model"
    ],

    "html": [
        "Portfolio Website",
        "Restaurant Website",
        "Landing Page Design"
    ],

    "css": [
        "Responsive Dashboard UI",
        "Animated Website",
        "Modern Portfolio Design"
    ],

    "javascript": [
        "Weather App",
        "Todo Application",
        "Quiz Application"
    ],

    "react": [
        "React Admin Dashboard",
        "Realtime Chat UI",
        "E-commerce Frontend"
    ],

    "nodejs": [
        "REST API Development",
        "Authentication System",
        "Realtime Chat Backend"
    ],

    "api": [
        "Weather API Project",
        "Movie Recommendation API",
        "AI Chatbot API"
    ],

    "database": [
        "Inventory Management System",
        "Banking Database System",
        "Student Record Management"
    ],

    "aws": [
        "Cloud Web Application",
        "Serverless Deployment",
        "AWS Portfolio Project"
    ],

    "docker": [
        "Dockerized Flask App",
        "Containerized ML Project",
        "Docker Deployment Pipeline"
    ],

    "kubernetes": [
        "Kubernetes Cluster Deployment",
        "Microservices Deployment",
        "Scalable Cloud Application"
    ],

    "linux": [
        "Linux Automation Scripts",
        "Server Monitoring System",
        "Linux User Management"
    ],

    "ci/cd": [
        "Automated Deployment Pipeline",
        "GitHub Actions Workflow",
        "Jenkins CI/CD Project"
    ]
}

# ---------------------------
# DEPENDENCIES
# ---------------------------
dependencies = {

    "machine learning": [
        "python"
    ],

    "deep learning": [
        "machine learning"
    ]
}

# ---------------------------
# STOPWORDS
# ---------------------------
stop_words = set(
    stopwords.words('english')
)

# ---------------------------
# SESSION STATE
# ---------------------------
defaults = {

    "analysis_done": False,
    "matched": [],
    "missing": [],
    "completed": [],
    "readiness": 0,
    "resume_score": 0,
    "user_skills": "",
    "chat_history": []
}

for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value

# ---------------------------
# PREPROCESS
# ---------------------------
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

# ---------------------------
# PDF EXTRACTION
# ---------------------------
def extract_pdf(pdf):

    reader = PyPDF2.PdfReader(pdf)

    text = ""

    for page in reader.pages:

        t = page.extract_text()

        if t:
            text += t

    return text

# ---------------------------
# SKILL EXTRACTION
# ---------------------------
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

# ---------------------------
# ANALYSIS
# ---------------------------
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

# ---------------------------
# SIMILARITY
# ---------------------------
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

# ---------------------------
# ROLE RECOMMENDATION
# ---------------------------
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

# ---------------------------
# ROADMAP SORTING
# ---------------------------
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

# ---------------------------
# RESOURCES
# ---------------------------
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

# ---------------------------
# OFFLINE CHATBOT
# ---------------------------
def real_ai_chat(user_query, role):

    query = user_query.lower()

    matched = st.session_state.matched
    missing = st.session_state.missing

    response = ""

    if "roadmap" in query:

        response = f"""
# 🚀 Career Roadmap for {role.replace('_',' ').title()}

## ✅ Your Current Skills
{', '.join(matched)}

## ❌ Skills To Learn
{', '.join(missing)}

"""

        for i, skill in enumerate(missing):

            response += f"\n### Step {i+1}: Learn {skill.title()}\n"

            if skill in project_recommendations:

                response += "\nRecommended Projects:\n"

                for p in project_recommendations[skill]:

                    response += f"- {p}\n"

    elif "project" in query:

        response = "# 💡 Recommended Projects\n"

        shown = set()

        for skill in matched + missing:

            if skill in project_recommendations:

                response += f"\n## {skill.title()}\n"

                for project in project_recommendations[skill]:

                    if project not in shown:

                        response += f"- {project}\n"
                        shown.add(project)

    elif "interview" in query:

        response = """
# 🎯 Interview Preparation

## Technical Round
- Practice coding
- Revise SQL
- Build projects

## HR Round
- Introduce yourself confidently
- Explain projects clearly
- Prepare strengths and weaknesses

## Important Tip
Communication + Projects = Higher Selection Chances
"""

    elif "data analyst" in query:

        response = """
# 📊 Data Analyst

A Data Analyst collects and analyzes data to help companies make decisions.

## Skills Required
- Python
- SQL
- Excel
- Statistics
- Power BI

## Recommended Projects
- Sales Dashboard
- Customer Analytics
- HR Analytics

## Career Advice
Master SQL and Excel first.
Then move to Python and visualization tools.
"""

    else:

        response = f"""
# 🤖 AI Career Mentor

I can help you with:

- Career Roadmaps
- Skill Guidance
- Project Recommendations
- Resume Tips
- Interview Preparation

Selected Role:
## {role.replace('_',' ').title()}

Try asking:
- Give roadmap
- Recommend projects
- Explain data analyst
- Interview tips
"""

    return response

# ---------------------------
# PDF REPORT
# ---------------------------
def generate_pdf(
    role,
    matched,
    missing,
    readiness
):

    pdf = FPDF()

    pdf.add_page()

    pdf.set_font(
        "Arial",
        size=12
    )

    pdf.cell(
        200,
        10,
        txt="AI Career Report",
        ln=True
    )

    pdf.cell(
        200,
        10,
        txt=f"Target Role: {role}",
        ln=True
    )

    pdf.cell(
        200,
        10,
        txt=f"Readiness: {readiness}%",
        ln=True
    )

    pdf.cell(
        200,
        10,
        txt="Matched Skills:",
        ln=True
    )

    for skill in matched:

        pdf.cell(
            200,
            8,
            txt=f"- {skill}",
            ln=True
        )

    pdf.cell(
        200,
        10,
        txt="Missing Skills:",
        ln=True
    )

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

# ---------------------------
# UI
# ---------------------------
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

    st.success(
        "Resume Processed Successfully"
    )

    st.write(
        "Extracted Skills:",
        auto_skills
    )

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

# ---------------------------
# RESULTS
# ---------------------------
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

    st.subheader(
        "🎯 Recommended Roles"
    )

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

        st.write(
            f"Match Score: {score}%"
        )

    with st.expander(
        "✅ Matched Skills"
    ):

        for skill in st.session_state.matched:

            st.success(skill)

    with st.expander(
        "❌ Missing Skills"
    ):

        for skill in st.session_state.missing:

            st.error(skill)

    st.subheader(
        "📚 Adaptive Learning Roadmap"
    )

    ordered = sort_skills(
        st.session_state.missing
    )

    for i, skill in enumerate(ordered):

        st.markdown(
            f"### Week {i*2+1}-{i*2+2}: {skill.title()}"
        )

        res = resources(skill)

        st.markdown(
            f"- [Course]({res['course']})"
        )

        st.markdown(
            f"- [Documentation]({res['docs']})"
        )

        st.markdown(
            f"- [Practice Projects]({res['practice']})"
        )

        st.write(
            "### Recommended Projects"
        )

        if skill in project_recommendations:

            for project in project_recommendations[skill]:

                st.write(f"• {project}")

        st.markdown("---")

    # ---------------------------
    # CHART
    # ---------------------------
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

    # ---------------------------
    # CHATBOT
    # ---------------------------
    st.subheader(
        "🤖 AI Career Mentor"
    )

    for chat in st.session_state.chat_history:

        with st.chat_message(
            chat["role"]
        ):

            st.markdown(
                chat["message"]
            )

    user_prompt = st.chat_input(
        "Ask your career question..."
    )

    if user_prompt:

        st.session_state.chat_history.append({

            "role": "user",
            "message": user_prompt
        })

        response = real_ai_chat(
            user_prompt,
            role
        )

        st.session_state.chat_history.append({

            "role": "assistant",
            "message": response
        })

        st.rerun()

    # ---------------------------
    # PDF REPORT
    # ---------------------------
    if st.button(
        "Generate PDF Report"
    ):

        file = generate_pdf(
            role,
            st.session_state.matched,
            st.session_state.missing,
            st.session_state.readiness
        )

        with open(file, "rb") as f:

            st.download_button(
                label="Download PDF Report",
                data=f,
                file_name="AI_Career_Report.pdf",
                mime="application/pdf"
            )