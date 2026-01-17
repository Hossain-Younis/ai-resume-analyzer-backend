import re
import spacy
from datetime import datetime

nlp = spacy.load("en_core_web_sm")

# INFO EXTRACTION
def extract_resume_info(text):
    doc = nlp(text)

    # Name (first PERSON entity)
    name = None
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break

    email = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    phone = re.search(r"\+?\d[\d\s\-]{8,}", text)

    skills_db = [
        "Python","Java","C++","TensorFlow","PyTorch","React","Node.js",
        "Docker","Kubernetes","AWS","SQL","Machine Learning",
        "Deep Learning","FastAPI","Django","Flask"
    ]

    skills_found = [s for s in skills_db if s.lower() in text.lower()]

    return {
        "name": name,
        "email": email.group(0) if email else None,
        "phone": phone.group(0) if phone else None,
        "skills": list(set(skills_found))
    }

# EXPERIENCE CALCULATION
def calculate_experience(text):
    date_ranges = re.findall(
        r"(20\d{2}|19\d{2})\s*[-–to]+\s*(20\d{2}|Present)",
        text,
        re.IGNORECASE
    )

    total_years = 0
    for start, end in date_ranges:
        start = int(start)
        end = datetime.now().year if end.lower() == "present" else int(end)
        total_years += max(0, end - start)

    return total_years