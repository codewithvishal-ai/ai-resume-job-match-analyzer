import re
from collections import Counter
from io import BytesIO

import streamlit as st

try:
    import docx
except ImportError:  # pragma: no cover - handled in UI
    docx = None

try:
    import fitz
except ImportError:  # pragma: no cover - handled in UI
    fitz = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="AI Resume Match Analyzer",
    page_icon="",
    layout="wide",
)


SKILL_BANK = {
    "Programming": [
        "python",
        "java",
        "c++",
        "c",
        "javascript",
        "typescript",
        "sql",
        "r",
        "html",
        "css",
    ],
    "AI and ML": [
        "machine learning",
        "deep learning",
        "nlp",
        "natural language processing",
        "computer vision",
        "tensorflow",
        "pytorch",
        "keras",
        "scikit-learn",
        "sklearn",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "opencv",
        "transformers",
        "llm",
        "generative ai",
    ],
    "Data": [
        "data analysis",
        "data visualization",
        "power bi",
        "tableau",
        "excel",
        "statistics",
        "data cleaning",
        "etl",
        "big data",
        "spark",
        "hadoop",
    ],
    "Backend and Cloud": [
        "flask",
        "fastapi",
        "django",
        "streamlit",
        "api",
        "rest api",
        "docker",
        "kubernetes",
        "aws",
        "azure",
        "gcp",
        "mongodb",
        "mysql",
        "postgresql",
        "sqlite",
        "git",
        "github",
    ],
    "Soft Skills": [
        "communication",
        "leadership",
        "teamwork",
        "problem solving",
        "analytical thinking",
        "presentation",
        "collaboration",
    ],
}


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
    "we",
    "you",
    "your",
    "will",
    "our",
    "their",
    "candidate",
    "experience",
    "role",
    "job",
    "work",
    "team",
}


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_pdf_text(file_bytes):
    if fitz is None:
        return "", "Install PyMuPDF to read PDF files."

    document = fitz.open(stream=BytesIO(file_bytes), filetype="pdf")
    pages = [page.get_text("text") for page in document]
    return "\n".join(pages), None


def extract_docx_text(file_bytes):
    if docx is None:
        return "", "Install python-docx to read DOCX files."

    document = docx.Document(BytesIO(file_bytes))
    paragraphs = [paragraph.text for paragraph in document.paragraphs]
    return "\n".join(paragraphs), None


def extract_resume_text(uploaded_file):
    if uploaded_file is None:
        return "", None

    file_bytes = uploaded_file.read()
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        return extract_pdf_text(file_bytes)
    if filename.endswith(".docx"):
        return extract_docx_text(file_bytes)
    if filename.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore"), None

    return "", "Unsupported file type. Upload a PDF, DOCX, or TXT file."


def flatten_skill_bank():
    skills = []
    for category_skills in SKILL_BANK.values():
        skills.extend(category_skills)
    return sorted(set(skills), key=len, reverse=True)


def extract_skills(text):
    normalized = normalize_text(text)
    found = set()

    for skill in flatten_skill_bank():
        pattern = r"(?<![a-z0-9+#])" + re.escape(skill) + r"(?![a-z0-9+#])"
        if re.search(pattern, normalized):
            found.add(skill)

    return sorted(found)


def extract_keywords(text, top_n=12):
    normalized = normalize_text(text)
    words = re.findall(r"[a-z][a-z0-9+#.-]{2,}", normalized)
    words = [word for word in words if word not in STOPWORDS]
    counts = Counter(words)
    return [word for word, _ in counts.most_common(top_n)]


def calculate_similarity(resume_text, job_text):
    if not resume_text.strip() or not job_text.strip():
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    vectors = vectorizer.fit_transform([resume_text, job_text])
    return float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0])


def calculate_ats_score(resume_text, job_text, resume_skills, job_skills):
    similarity_score = calculate_similarity(resume_text, job_text) * 100
    if job_skills:
        skill_score = (len(set(resume_skills) & set(job_skills)) / len(job_skills)) * 100
    else:
        skill_score = 0

    keyword_overlap = set(extract_keywords(resume_text, 30)) & set(extract_keywords(job_text, 30))
    keyword_score = min(len(keyword_overlap) * 5, 100)

    final_score = (similarity_score * 0.45) + (skill_score * 0.4) + (keyword_score * 0.15)
    return round(final_score, 1), round(similarity_score, 1), round(skill_score, 1), round(keyword_score, 1)


def get_resume_sections(text):
    normalized = normalize_text(text)
    sections = {
        "summary": bool(re.search(r"\b(summary|objective|profile)\b", normalized)),
        "skills": bool(re.search(r"\b(skills|technical skills|technologies)\b", normalized)),
        "projects": bool(re.search(r"\b(projects|academic projects|personal projects)\b", normalized)),
        "experience": bool(re.search(r"\b(experience|internship|work history)\b", normalized)),
        "education": bool(re.search(r"\b(education|degree|university|college)\b", normalized)),
    }
    return sections


def build_suggestions(score, missing_skills, sections, resume_keywords, job_keywords):
    suggestions = []

    if score < 60:
        suggestions.append("Add more job-specific skills and keywords from the job description.")
    elif score < 80:
        suggestions.append("Your resume is close. Improve it by adding measurable project results.")
    else:
        suggestions.append("Strong match. Polish formatting and keep the most relevant projects near the top.")

    for section, present in sections.items():
        if not present:
            suggestions.append(f"Add a clear {section.title()} section.")

    missing_keywords = [keyword for keyword in job_keywords if keyword not in resume_keywords]
    if missing_keywords:
        suggestions.append("Include important keywords where truthful: " + ", ".join(missing_keywords[:6]) + ".")

    if missing_skills:
        suggestions.append("Prioritize learning or demonstrating: " + ", ".join(missing_skills[:6]) + ".")

    suggestions.append("Use action verbs and numbers, such as improved accuracy by 12% or built 3 ML models.")
    return suggestions[:8]


def generate_interview_questions(job_skills, missing_skills, job_keywords):
    focus_skills = list(dict.fromkeys(job_skills[:5] + missing_skills[:5]))
    questions = []

    for skill in focus_skills[:6]:
        questions.append(f"How have you used {skill} in a project or academic work?")
        questions.append(f"What are the common challenges while working with {skill}?")

    for keyword in job_keywords[:4]:
        questions.append(f"Can you explain your understanding of {keyword} in this role?")

    questions.extend(
        [
            "Tell me about one project where you solved a real problem using data or AI.",
            "How do you evaluate whether a machine learning model is performing well?",
            "Describe a time when you learned a new technology quickly.",
            "Why are you interested in this role and company?",
        ]
    )

    return questions[:12]


def score_label(score):
    if score >= 80:
        return "Excellent match"
    if score >= 60:
        return "Good match"
    if score >= 40:
        return "Moderate match"
    return "Needs improvement"


def render_skill_group(title, skills):
    if not skills:
        st.caption("No skills found yet.")
        return

    st.markdown(
        " ".join([f"`{skill}`" for skill in skills])
    )


st.title("AI Resume + Job Match Analyzer")
st.caption("Upload a resume, paste a job description, and get an ATS score, skill gap analysis, and interview questions.")

with st.sidebar:
    st.header("How to Use")
    st.write("1. Upload your resume.")
    st.write("2. Paste a job description.")
    st.write("3. Click Analyze Match.")
    st.divider()
    st.write("Supported resume formats: PDF, DOCX, TXT")

left, right = st.columns([1, 1])

with left:
    uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])
    resume_text_manual = st.text_area(
        "Or paste resume text",
        height=180,
        placeholder="Paste your resume text here if you do not want to upload a file.",
    )

with right:
    job_description = st.text_area(
        "Paste Job Description",
        height=300,
        placeholder="Paste the job description here...",
    )

analyze = st.button("Analyze Match", type="primary", use_container_width=True)

if analyze:
    extracted_resume_text, error = extract_resume_text(uploaded_file)
    resume_text = resume_text_manual.strip() or extracted_resume_text.strip()

    if error:
        st.error(error)
    elif not resume_text:
        st.warning("Please upload a resume or paste resume text.")
    elif not job_description.strip():
        st.warning("Please paste a job description.")
    else:
        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_description)
        matched_skills = sorted(set(resume_skills) & set(job_skills))
        missing_skills = sorted(set(job_skills) - set(resume_skills))
        resume_keywords = extract_keywords(resume_text)
        job_keywords = extract_keywords(job_description)
        sections = get_resume_sections(resume_text)

        ats_score, similarity_score, skill_score, keyword_score = calculate_ats_score(
            resume_text,
            job_description,
            resume_skills,
            job_skills,
        )

        st.divider()
        st.subheader("Match Report")

        metric_1, metric_2, metric_3, metric_4 = st.columns(4)
        metric_1.metric("ATS Score", f"{ats_score}%")
        metric_2.metric("Semantic Similarity", f"{similarity_score}%")
        metric_3.metric("Skill Match", f"{skill_score}%")
        metric_4.metric("Keyword Match", f"{keyword_score}%")

        st.progress(min(int(ats_score), 100), text=score_label(ats_score))

        tab_summary, tab_skills, tab_tips, tab_questions = st.tabs(
            ["Summary", "Skill Gap", "Suggestions", "Interview Questions"]
        )

        with tab_summary:
            st.write(f"Overall result: **{score_label(ats_score)}**")
            st.write(f"Resume words analyzed: **{len(resume_text.split())}**")
            st.write(f"Job description words analyzed: **{len(job_description.split())}**")
            st.write("Top job keywords:")
            render_skill_group("Job Keywords", job_keywords)

            st.write("Resume section check:")
            for section, present in sections.items():
                st.write(f"- {section.title()}: {'Found' if present else 'Missing'}")

        with tab_skills:
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.write("Matched Skills")
                render_skill_group("Matched Skills", matched_skills)
            with col_b:
                st.write("Missing Skills")
                render_skill_group("Missing Skills", missing_skills)
            with col_c:
                st.write("Skills Found in Resume")
                render_skill_group("Resume Skills", resume_skills)

        with tab_tips:
            suggestions = build_suggestions(
                ats_score,
                missing_skills,
                sections,
                resume_keywords,
                job_keywords,
            )
            for index, suggestion in enumerate(suggestions, start=1):
                st.write(f"{index}. {suggestion}")

        with tab_questions:
            questions = generate_interview_questions(job_skills, missing_skills, job_keywords)
            for index, question in enumerate(questions, start=1):
                st.write(f"{index}. {question}")

        report = [
            "AI Resume + Job Match Analyzer Report",
            "",
            f"ATS Score: {ats_score}%",
            f"Overall Result: {score_label(ats_score)}",
            "",
            "Matched Skills:",
            ", ".join(matched_skills) if matched_skills else "None found",
            "",
            "Missing Skills:",
            ", ".join(missing_skills) if missing_skills else "None found",
            "",
            "Suggestions:",
            *[f"- {item}" for item in build_suggestions(ats_score, missing_skills, sections, resume_keywords, job_keywords)],
            "",
            "Interview Questions:",
            *[f"{index}. {question}" for index, question in enumerate(generate_interview_questions(job_skills, missing_skills, job_keywords), start=1)],
        ]

        st.download_button(
            "Download Report",
            data="\n".join(report),
            file_name="resume_match_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
else:
    st.info("Add your resume and job description, then click Analyze Match.")
