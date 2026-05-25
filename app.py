import streamlit as st
import random
import re

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="AI Career Planner",
    page_icon="🤖",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
        color: white;
    }

    .stTextInput > div > div > input {
        background-color: #1e293b;
        color: white;
        border-radius: 10px;
    }

    .chat-user {
        background-color: #2563eb;
        padding: 12px;
        border-radius: 12px;
        margin: 10px 0;
        color: white;
    }

    .chat-bot {
        background-color: #1e293b;
        padding: 12px;
        border-radius: 12px;
        margin: 10px 0;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# TITLE
# ==========================================
st.title("🤖 AI Career Planner")
st.subheader("Offline AI Mentor Chatbot + Project Recommendation System")

# ==========================================
# SESSION STATE
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# CAREER ROADMAP DATA
# ==========================================
career_data = {
    "data analyst": {
        "skills": [
            "Python",
            "SQL",
            "Excel",
            "Power BI",
            "Tableau",
            "Statistics",
            "Pandas"
        ],
        "projects": [
            "Sales Dashboard",
            "Customer Churn Analysis",
            "COVID-19 Data Analysis",
            "E-commerce Analytics",
            "HR Analytics Dashboard"
        ],
        "advice": "Focus on data cleaning, visualization, and SQL queries."
    },

    "web developer": {
        "skills": [
            "HTML",
            "CSS",
            "JavaScript",
            "React",
            "Node.js",
            "MongoDB",
            "Git"
        ],
        "projects": [
            "Portfolio Website",
            "E-commerce Website",
            "Blog Platform",
            "Weather App",
            "Task Manager"
        ],
        "advice": "Build responsive projects and improve frontend + backend integration."
    },

    "ai engineer": {
        "skills": [
            "Python",
            "Machine Learning",
            "Deep Learning",
            "TensorFlow",
            "NLP",
            "Computer Vision"
        ],
        "projects": [
            "AI Chatbot",
            "Resume Analyzer",
            "Face Detection System",
            "Sentiment Analysis",
            "Recommendation System"
        ],
        "advice": "Focus on ML algorithms and real-world AI applications."
    },

    "cybersecurity": {
        "skills": [
            "Networking",
            "Ethical Hacking",
            "Linux",
            "Cryptography",
            "Python",
            "Security Testing"
        ],
        "projects": [
            "Password Strength Checker",
            "Port Scanner",
            "Keylogger Detector",
            "Network Monitoring Tool",
            "Cybersecurity Dashboard"
        ],
        "advice": "Practice ethical hacking labs and network security concepts."
    },

    "cloud engineer": {
        "skills": [
            "AWS",
            "Azure",
            "Docker",
            "Kubernetes",
            "Linux",
            "DevOps"
        ],
        "projects": [
            "Cloud Deployment System",
            "Dockerized Web App",
            "CI/CD Pipeline",
            "AWS Storage Manager",
            "Cloud Monitoring Dashboard"
        ],
        "advice": "Learn deployment and cloud infrastructure management."
    }
}

# ==========================================
# CHATBOT RESPONSE FUNCTION
# ==========================================
def offline_chatbot(user_input):

    text = user_input.lower()

    # Greetings
    greetings = ["hi", "hello", "hey", "good morning"]

    if any(word in text for word in greetings):
        return random.choice([
            "Hello 👋 How can I help you today?",
            "Hi there! Ask me anything about careers or projects.",
            "Welcome 🚀 Tell me your career goal."
        ])

    # Career Roadmap
    for role in career_data:
        if role in text:

            data = career_data[role]

            response = f"## 🚀 Career Role: {role.title()}\n\n"

            response += "### 🔥 Skills Required\n"
            for skill in data["skills"]:
                response += f"- {skill}\n"

            response += "\n### 💡 Recommended Projects\n"
            for project in data["projects"]:
                response += f"- {project}\n"

            response += f"\n### 📌 Advice\n{data['advice']}"

            return response

    # Skills Questions
    if "skills" in text:
        return "Important skills are Python, SQL, AI, Web Development, Communication, and Problem Solving."

    # Resume Questions
    if "resume" in text:
        return "A strong resume should include skills, projects, internships, certifications, and achievements."

    # Interview Questions
    if "interview" in text:
        return "Practice aptitude, communication, projects, and technical coding questions for interviews."

    # Project Questions
    if "project" in text:
        return "You can build AI Chatbots, Resume Analyzers, Recommendation Systems, Dashboards, and Web Apps."

    # Python Questions
    if "python" in text:
        return "Python is widely used in AI, data science, automation, and web development."

    # SQL Questions
    if "sql" in text:
        return "SQL is used to manage and analyze data stored in databases."

    # AI Questions
    if "ai" in text:
        return "Artificial Intelligence helps machines learn, analyze, and make decisions automatically."

    # Default Response
    return random.choice([
        "Can you explain your question in more detail?",
        "Interesting question 👀",
        "I can help with careers, skills, projects, AI, Python, and interviews.",
        "Please ask about career guidance or projects."
    ])

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:

    st.header("📚 Career Roles")

    st.write("- Data Analyst")
    st.write("- Web Developer")
    st.write("- AI Engineer")
    st.write("- Cybersecurity")
    st.write("- Cloud Engineer")

    st.divider()

    st.header("🛠 Features")

    st.write("✅ AI Mentor Chatbot")
    st.write("✅ Career Roadmaps")
    st.write("✅ Skills Recommendation")
    st.write("✅ Project Recommendation")
    st.write("✅ Offline Chatbot")

# ==========================================
# CHAT DISPLAY
# ==========================================
for message in st.session_state.messages:

    if message["role"] == "user":
        st.markdown(
            f'<div class="chat-user">🧑 {message["content"]}</div>',
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            f'<div class="chat-bot">🤖 {message["content"]}</div>',
            unsafe_allow_html=True
        )

# ==========================================
# CHAT INPUT
# ==========================================
user_question = st.chat_input("Ask your question...")

# ==========================================
# PROCESS USER MESSAGE
# ==========================================
if user_question:

    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    # Generate bot response
    bot_response = offline_chatbot(user_question)

    # Store bot response
    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_response
    })

    # Rerun app
    st.rerun()


st.divider()
st.caption("🚀 AI Career Planner | Offline AI Chatbot | Built with Streamlit")
```


