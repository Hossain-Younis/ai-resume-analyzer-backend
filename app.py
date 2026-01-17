from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pickle, json
import fitz  # PyMuPDF
from utils import extract_resume_info, calculate_experience
import numpy as np

app = FastAPI(title="AI Resume Analyzer")

# CORS (for React frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML assets
with open("resume_classifier_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("resume_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("categories.json") as f:
    categories = json.load(f)


# PDF → TEXT
def pdf_to_text(file: UploadFile):
    doc = fitz.open(stream=file.file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text



# API ENDPOINT
@app.post("/analyze")
async def analyze_resume(file: UploadFile = File(None), text: str = None):

    if file:
        resume_text = pdf_to_text(file)
    elif text:
        resume_text = text
    else:
        return {"error": "No resume provided"}

    # Extraction
    extracted = extract_resume_info(resume_text)

    # Experience calculation
    experience_years = calculate_experience(resume_text)

    if experience_years < 2:
        level = "Junior"
    elif experience_years < 5:
        level = "Mid"
    else:
        level = "Senior"

    # Classification
    tfidf = vectorizer.transform([resume_text])
    prediction = model.predict(tfidf)[0]
    probs = model.predict_proba(tfidf)[0]
    confidence = float(np.max(probs))

    return {
        "name": extracted["name"],
        "email": extracted["email"],
        "phone": extracted["phone"],
        "skills": extracted["skills"],
        "experience_years": round(experience_years, 2),
        "experience_level": level,
        "classification": prediction,
        "confidence": round(confidence, 2)
    }