from flask import Flask, render_template, request
from openai import OpenAI
from PyPDF2 import PdfReader
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from dotenv import load_dotenv
import uuid
import os

filename = f"{uuid.uuid4()}.pdf"

load_dotenv()

app = Flask(__name__)

# NVIDIA NIM Client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

# Home Page
@app.route("/")
def home():
    return render_template("index.html")


# Analyze Route
@app.route("/analyze", methods=["POST"])
def analyze():

    resume_file = request.files["resume"]
    job_description = request.form["job_description"]
    experience_level = request.form["experience_level"]
    generate_cover_letter = request.form.get("generate_cover_letter")

    # Read the resume file
    if resume_file:
        pdf_reader = PdfReader(resume_file)
        resume_text = ""
        for page in pdf_reader.pages:
            resume_text += page.extract_text()

    prompt = f"""
You are an ATS Resume Analyzer.

Analyze the resume against the job description.

Resume:
{resume_text}

Job Description:
{job_description}

Experience Level:
{experience_level}

Provide:

1. Match Percentage

2. Matching Skills

3. Missing Skills

4. Suggestions for Improvement

5. Top 3 strengths

6. Top 3 weaknesses

Return the result EXACTLY in this format:

MATCH PERCENTAGE:
<value>

MATCHING SKILLS:
- Skill 1
- Skill 2
- Skill 3

MISSING SKILLS:
- Skill 1
- Skill 2

SUGGESTIONS:
- Suggestion 1
- Suggestion 2

STRENGTHS:
1. Strength 1   
2. Strength 2
3. Strength 3
WEAKNESSES:
1. Weakness 1
2. Weakness 2
3. Weakness 3

Use line breaks between every section.
Do not write everything in one paragraph.

Format the response clearly with headings.
"""
    if generate_cover_letter:
        prompt += """
Use EXACTLY this heading:

COVER LETTER:
Also generate a professional cover letter tailored to this job description and candidate.
"""

    response = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",  # replace if you're using a different model
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=1000
    )

    result = response.choices[0].message.content
    styles = getSampleStyleSheet()
    analysis_text = result.strip()
    cover_letter_text = ""

    if generate_cover_letter:
        for marker in ("Cover Letter:", "COVER LETTER:"):
            if marker in result:
                before, sep, after = result.partition(marker)
                analysis_text = before.strip()
                cover_letter_text = after.strip()
                break

    content = [
        Paragraph("Analysis", styles["Heading2"]),
        Paragraph(analysis_text.replace("\n", "<br/>"), styles["Normal"])
    ]
    if cover_letter_text:
        content.append(Paragraph("Cover Letter", styles["Heading2"]))
        content.append(Paragraph(cover_letter_text.replace("\n", "<br/>"), styles["Normal"]))
    # Create a PDF document

    from flask import send_file
    return send_file(
    filename,
    as_attachment=True
)



if __name__ == "__main__":
    app.run(debug=True)
