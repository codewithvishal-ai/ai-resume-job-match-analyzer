# AI Resume + Job Match Analyzer

A Python-only Streamlit web app that compares a resume with a job description and generates:

- ATS-style match score
- Semantic similarity score
- Skill gap analysis
- Resume improvement suggestions
- Personalized interview questions
- Downloadable text report

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown in the terminal.

## Supported Inputs

- Resume upload: PDF, DOCX, TXT
- Manual resume text paste
- Job description text paste

## Project Modules

- Resume text extraction
- Skill extraction
- TF-IDF similarity
- ATS score calculation
- Missing skill detection
- Resume suggestions
- Interview question generation
