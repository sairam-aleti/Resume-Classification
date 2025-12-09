# resumes_classification.py
#
# Streamlit app for:
# - Classifying resumes into tech roles
# - Extracting contact info (name, phone, email)
# - Ranking candidates for a given Job Description (JD)
# - Managing a candidate pool, persisted in MongoDB
#
# Required files:
#   - best_role_classifier.joblib (trained scikit-learn pipeline)
#   - data.py (MongoDB repository)
#
# Environment variables (MongoDB):
#   - MONGO_URI: MongoDB connection string 
#     (default: mongodb://localhost:27017)
#   - MONGO_DB: Database name (default: resume_classifier)
#   - MONGO_COLLECTION: Collection name (default: candidates)
#
# Install dependencies:
#   pip install streamlit pdfplumber scikit-learn joblib pandas numpy pymongo

import re
import traceback
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import vstack
from sklearn.metrics.pairwise import linear_kernel

from data import get_candidate_repository

# Optional PDF support
try:
    import pdfplumber

    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False


# =========================
# Name & contact extraction
# =========================

phone_re = re.compile(r"(\+?\d[\d\s\-]{9,})")
email_re = re.compile(r"[\w\.-]+@[\w\.-]+")

BAD_NAME_KEYWORDS = {
    "http",
    "www",
    "github",
    "gitlab",
    "linkedin",
    "stackoverflow",
    "resume",
    "curriculum vitae",
    "curriculum-vitae",
    "cv",
    "email",
    "phone",
    "mobile",
    "@",
    "summary",
    "professional summary",
    "objective",
    "career objective",
    "profile",
    "professional profile",
    "experience",
    "work experience",
    "employment history",
    "skills",
    "technical skills",
    "key skills",
    "education",
    "projects",
    "certifications",
    "publications",
    "interests",
    "hobbies",
}


def _line_has_bad_keywords(line: str) -> bool:
    low = line.lower()
    low_clean = low.rstrip(":").strip()
    if low_clean in BAD_NAME_KEYWORDS:
        return True
    if any(k in low for k in BAD_NAME_KEYWORDS):
        return True
    return False


def guess_name_from_email_local(local: str):
    """
    Heuristic split of an email local-part into spaced tokens using
    a list of common name segments.

    Example:
      'sairamreddyaleti' -> 'Sai Ram Reddy Aleti'
      'alexkumar'        -> 'Alex Kumar'
    """
    if not local:
        return None
    local = re.sub(r"\d+", "", local.lower())
    if not local:
        return None

    segments = [
        "sairam",
        "sai",
        "ram",
        "reddy",
        "aleti",
        "kumar",
        "karthik",
        "karthi",
        "singh",
        "sharma",
        "patel",
        "khan",
        "gupta",
        "iyer",
        "joshi",
        "das",
        "nair",
        "alex",
        "sam",
        "priya",
        "rohan",
        "meera",
        "anita",
    ]
    segments.sort(key=len, reverse=True)

    tokens = []
    i = 0
    while i < len(local):
        matched = None
        for seg in segments:
            if local.startswith(seg, i):
                matched = seg
                break
        if matched:
            tokens.append(matched.capitalize())
            i += len(matched)
        else:
            chunk = local[i : i + 4]
            tokens.append(chunk.capitalize())
            i += len(chunk)
        if len(tokens) >= 4:
            break

    return " ".join(tokens)


def infer_name_from_email_and_text(text: str, email: str):
    """
    Use email local-part and resume text together to infer a spaced name.

    - Get email local-part, e.g. 'sairamreddyaleti'
    - From first ~15 non-empty lines, collect tokens whose letters are
      substrings of local-part (ignoring digits, URLs, etc.).
    - Order tokens by their position in the local-part and join them (up to 4 tokens).
      Example:
        local = 'sairamreddyaleti'
        tokens in text: 'Sai', 'Ram', 'Reddy', 'Aleti', 'Analytics', ...
        -> picks 'Sai Ram Reddy Aleti'
    """
    if not email:
        return None
    local = email.split("@", 1)[0]
    local = re.sub(r"\d+", "", local.lower())
    if not local:
        return None

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    candidates = []
    seen = set()

    for line in lines[:15]:
        if _line_has_bad_keywords(line):
            continue
        for tok in line.split():
            alph = "".join(ch for ch in tok if ch.isalpha())
            if len(alph) < 2:
                continue
            low_tok = alph.lower()
            if low_tok in seen:
                continue
            if "@" in tok:
                continue
            if any(ch.isdigit() for ch in tok):
                continue
            if any(x in low_tok for x in ("http", "www", "github", "linkedin")):
                continue
            pos = local.find(low_tok)
            if pos != -1:
                candidates.append((pos, alph))
                seen.add(low_tok)

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])

    selected = []
    last_end = -1
    for pos, alph in candidates:
        end = pos + len(alph)
        if pos >= last_end:
            selected.append(alph)
            last_end = end
            if len(selected) >= 4:
                break

    if len(selected) >= 2:
        concat = "".join(w.lower() for w in selected)
        if len(concat) / max(1, len(local)) >= 0.6:
            return " ".join(w.capitalize() for w in selected)

    return None


def extract_contacts(text: str):
    """Extract (name, phone, email) from resume text."""
    emails = email_re.findall(text)
    phones = phone_re.findall(text)
    email = emails[0] if emails else None
    phone = phones[0] if phones else None

    name = None

    if email:
        name = infer_name_from_email_and_text(text, email)
        if not name:
            local = email.split("@", 1)[0]
            name = guess_name_from_email_local(local)

    if not name:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        for line in lines[:10]:
            low = line.lower()
            if low.startswith("name:"):
                possible = line.split(":", 1)[1].strip()
                if possible:
                    name = possible
                    break

        if not name:
            for line in lines[:10]:
                if _line_has_bad_keywords(line):
                    continue
                if "@" in line or any(ch.isdigit() for ch in line):
                    continue
                name = line
                break

    return name, phone, email


# =========================
# Other helpers
# =========================

def parse_skills_cell(x):
    """Parse the 'skills' column from resumes.csv into a Python list."""
    import ast

    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    s = str(x).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            return [str(v).strip() for v in parsed]
        except Exception:
            pass
    if ";" in s:
        return [t.strip() for t in s.split(";") if t.strip()]
    if "," in s:
        return [t.strip() for t in s.split(",") if t.strip()]
    return [s]


def extract_skills_from_resume_text(text: str, vocab):
    txt = text.lower()
    found = set()
    for skill in vocab:
        if skill in txt:
            found.add(skill)
    return list(found)


def compute_experience_match(cand_years, min_exp, max_exp):
    cand = float(cand_years)
    if min_exp is None and max_exp is None:
        return 1.0
    if min_exp is None:
        min_exp = 0
    if max_exp is None:
        max_exp = min_exp + 5

    if cand <= 0:
        return 0.0
    if cand < min_exp:
        return max(0.0, cand / float(min_exp))
    if cand > max_exp:
        return max(0.0, float(max_exp) / cand)
    return 1.0


def compute_role_match(candidate_role, jd_role):
    if jd_role is None:
        return 0.5
    return 1.0 if candidate_role == jd_role else 0.0


def compute_skill_overlap(cand_skills, jd_skills):
    if not jd_skills:
        return 0.0
    cand_set = set(s.lower() for s in cand_skills)
    jd_set = set(s.lower() for s in jd_skills)
    overlap = cand_set & jd_set
    return len(overlap) / float(len(jd_set))


ROLE_KEYWORDS = {
    "Software Engineer": [
        "software engineer",
        "software developer",
        "application developer",
    ],
    "ML Engineer": [
        "ml engineer",
        "machine learning engineer",
        "ml ops",
    ],
    "Data Scientist": [
        "data scientist",
        "data science",
        "ml data scientist",
    ],
    "Backend / Full-Stack Developer": [
        "backend developer",
        "back-end developer",
        "full stack",
        "full-stack",
        "backend engineer",
    ],
    "DevOps / Cloud Engineer": [
        "devops",
        "cloud engineer",
        "site reliability engineer",
        "sre",
    ],
}

exp_pattern = re.compile(r"(\d+)\s*\+?\s*(?:years|year|yrs)\b", re.IGNORECASE)


def infer_role_from_jd(jd_text: str, predict_role_fn):
    text = jd_text.lower()
    for role, keywords in ROLE_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return role
    try:
        return predict_role_fn(jd_text)
    except Exception:
        return None


def extract_skills_from_jd(jd_text: str, skills_vocab):
    text = jd_text.lower()
    found = set()
    for skill in skills_vocab:
        if skill in text:
            found.add(skill)
    return found


def extract_experience_from_jd(jd_text: str):
    matches = exp_pattern.findall(jd_text)
    if not matches:
        return None, None
    years = sorted(int(m) for m in matches)
    if len(years) == 1:
        min_years = years[0]
        max_years = years[0] + 3
    else:
        min_years = years[0]
        max_years = years[-1]
    return min_years, max_years


def parse_jd(jd_text: str, skills_vocab, predict_role_fn):
    role = infer_role_from_jd(jd_text, predict_role_fn)
    skills = extract_skills_from_jd(jd_text, skills_vocab)
    min_exp, max_exp = extract_experience_from_jd(jd_text)
    return {
        "text": jd_text,
        "role": role,
        "skills": skills,
        "min_experience": min_exp,
        "max_experience": max_exp,
    }


# =========================
# Model + data loading
# =========================

def load_model():
    try:
        clf = joblib.load("best_role_classifier.joblib")
    except Exception as e:
        st.error(
            "Could not load best_role_classifier.joblib. "
            f"Make sure it exists in the same folder.\nError: {e}"
        )
        st.stop()
    if "tfidf" not in clf.named_steps:
        st.error("The loaded model pipeline does not have a 'tfidf' step.")
        st.stop()
    return clf


def init_app_state():
    if st.session_state.get("initialized"):
        return

    role_clf = load_model()

    # Load from persistent storage (MongoDB)
    try:
        repo = get_candidate_repository()
        df = repo.load_all()
    except Exception as e:
        st.error(
            f"Could not load candidate pool. "
            f"Check database connection and configuration.\nError: {e}"
        )
        st.stop()

    if len(df) == 0:
        st.warning("No candidates in MongoDB yet. Add some resumes to get started!")
        df = pd.DataFrame(columns=[
            "id", "name", "phone", "email", "predicted_role",
            "skills_list", "experience_years_num", "raw_text"
        ])

    if "raw_text" not in df.columns:
        st.error("Candidate pool must contain a 'raw_text' column.")
        st.stop()

    df["raw_text"] = df["raw_text"].astype(str)

    # Skills - MongoDB already stores as list, so just ensure it's a list
    if "skills_list" not in df.columns:
        df["skills_list"] = [[] for _ in range(len(df))]
    else:
        # Ensure all values are lists (in case of corrupted data)
        df["skills_list"] = df["skills_list"].apply(
            lambda x: x if isinstance(x, list) else []
        )

    # Experience - MongoDB stores as experience_years_num directly
    if "experience_years_num" not in df.columns:
        df["experience_years_num"] = 0.0
    else:
        df["experience_years_num"] = pd.to_numeric(
            df["experience_years_num"], errors="coerce"
        ).fillna(0.0)

    # Contacts - ensure they exist
    for col in ["phone", "email", "name"]:
        if col not in df.columns:
            df[col] = None
        else:
            df[col] = df[col].fillna("")

    # predicted_role - MongoDB stores this directly
    if "predicted_role" not in df.columns:
        # Fallback: predict if not in database
        df["predicted_role"] = role_clf.predict(df["raw_text"])

    # skills vocab
    all_skills = set()
    for skills in df["skills_list"]:
        if isinstance(skills, list):
            for s in skills:
                all_skills.add(str(s).lower())
    skills_vocab = sorted(all_skills)

    tfidf = role_clf.named_steps["tfidf"]
    resume_vectors = tfidf.transform(df["raw_text"])

    st.session_state["initialized"] = True
    st.session_state["role_clf"] = role_clf
    st.session_state["candidate_df"] = df
    st.session_state["skills_vocab"] = skills_vocab
    st.session_state["tfidf"] = tfidf
    st.session_state["resume_vectors"] = resume_vectors
    st.session_state["repo"] = repo


# =========================
# High-level functions
# =========================

def predict_role_from_text(text: str) -> str:
    clf = st.session_state["role_clf"]
    return clf.predict([text])[0]


def classify_resume_text(text: str):
    predicted_role = predict_role_from_text(text)
    name, phone, email = extract_contacts(text)
    return {
        "predicted_role": predicted_role,
        "name": name,
        "phone": phone,
        "email": email,
    }


def recompute_skills_vocab(df: pd.DataFrame):
    all_skills = set()
    if "skills_list" in df.columns:
        for skills in df["skills_list"]:
            if isinstance(skills, list):
                for s in skills:
                    all_skills.add(str(s).lower())
    return sorted(all_skills)


def check_duplicate_candidate(df: pd.DataFrame, name: str, phone: str, email: str, predicted_role: str):
    """
    Check if a candidate with the same name, phone, email, and role already exists.
    Returns (is_duplicate, duplicate_info) tuple.
    """
    if df.empty:
        return False, None
    
    # Normalize for comparison
    name_norm = (name or "").lower().strip() if name else ""
    phone_norm = (phone or "").lower().strip() if phone else ""
    email_norm = (email or "").lower().strip() if email else ""
    role_norm = (predicted_role or "").lower().strip() if predicted_role else ""
    
    for idx, row in df.iterrows():
        existing_name = (row.get("name") or "").lower().strip() if row.get("name") else ""
        existing_phone = (row.get("phone") or "").lower().strip() if row.get("phone") else ""
        existing_email = (row.get("email") or "").lower().strip() if row.get("email") else ""
        existing_role = (row.get("predicted_role") or "").lower().strip() if row.get("predicted_role") else ""
        
        # Check if all key fields match
        if (name_norm and existing_name and name_norm == existing_name and
            phone_norm and existing_phone and phone_norm == existing_phone and
            email_norm and existing_email and email_norm == existing_email and
            role_norm == existing_role):
            return True, {
                "id": row.get("id"),
                "name": row.get("name"),
                "email": row.get("email"),
                "phone": row.get("phone"),
                "role": row.get("predicted_role")
            }
    
    return False, None


def add_resume_text_to_pool(text: str):
    """
    Add a new resume (raw text) into MongoDB candidate pool.
    Includes duplicate detection.
    Recomputes TF-IDF vectors for ranking and updates session state.
    """
    repo = st.session_state.get("repo")

    if not repo:
        repo = get_candidate_repository()
        st.session_state["repo"] = repo

    try:
        # Step 1: Parse and classify resume
        parsed = classify_resume_text(text)
        predicted_role = parsed["predicted_role"]
        name = parsed["name"]
        phone = parsed["phone"]
        email = parsed["email"]
        
        # Step 2: Check for duplicates
        df = st.session_state["candidate_df"]
        is_duplicate, dup_info = check_duplicate_candidate(df, name, phone, email, predicted_role)
        
        if is_duplicate:
            return {
                "status": "duplicate",
                "existing_id": dup_info["id"],
                "existing_info": dup_info
            }

        # Step 3: Extract skills
        skills_vocab = st.session_state["skills_vocab"]
        skills_list = extract_skills_from_resume_text(text, skills_vocab)
        exp_years = 0.0

        # Step 4: Create candidate object
        new_id = str(uuid4())
        new_candidate = {
            "id": new_id,
            "name": name,
            "phone": phone,
            "email": email,
            "predicted_role": predicted_role,
            "skills_list": skills_list,
            "experience_years_num": exp_years,
            "raw_text": text,
        }

        # Step 5: Save to MongoDB directly
        result = repo.add_candidate(new_candidate)
        
        # Step 6: Reload from MongoDB to verify and update session state
        df = repo.load_all()
        
        # Step 7: Recompute skills vocab
        skills_vocab = recompute_skills_vocab(df)
        
        # Step 8: Recompute TF-IDF vectors
        tfidf = st.session_state["tfidf"]
        resume_vectors = tfidf.transform(df["raw_text"])

        # Step 9: Update all session state
        st.session_state["candidate_df"] = df
        st.session_state["skills_vocab"] = skills_vocab
        st.session_state["resume_vectors"] = resume_vectors

        return {"status": "success", "new_id": new_id}
        
    except Exception as e:
        error_msg = f"{str(e)} | {traceback.format_exc()}"
        return {"status": "error", "error_msg": error_msg}


def rank_candidates_for_jd(jd_text: str, top_k: int = 10):
    df = st.session_state["candidate_df"]
    skills_vocab = st.session_state["skills_vocab"]
    tfidf = st.session_state["tfidf"]
    resume_vectors = st.session_state["resume_vectors"]

    jd_info = parse_jd(jd_text, skills_vocab, predict_role_from_text)
    jd_role = jd_info["role"]
    jd_skills = jd_info["skills"]
    min_exp = jd_info["min_experience"]
    max_exp = jd_info["max_experience"]

    jd_vec = tfidf.transform([jd_text])
    cosine_similarities = linear_kernel(jd_vec, resume_vectors).flatten()

    scores = []
    for idx, row in df.iterrows():
        sem_sim = float(cosine_similarities[idx])
        skill_ov = compute_skill_overlap(row["skills_list"], jd_skills)
        exp_match = compute_experience_match(
            row["experience_years_num"], min_exp, max_exp
        )
        role_m = compute_role_match(row["predicted_role"], jd_role)

        final_score = (
            0.5 * sem_sim
            + 0.2 * skill_ov
            + 0.2 * exp_match
            + 0.1 * role_m
        )

        scores.append(
            {
                "id": row.get("id", idx),
                "name": row.get("name", None),
                "phone": row.get("phone", None),
                "email": row.get("email", None),
                "predicted_role": row.get("predicted_role", None),
                "semantic_similarity": sem_sim,
                "skill_overlap": skill_ov,
                "experience_match": exp_match,
                "role_match": role_m,
                "final_score": final_score,
            }
        )

    results_df = (
        pd.DataFrame(scores)
        .sort_values("final_score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    return results_df, jd_info


def pdf_to_text(uploaded_file) -> str:
    if not PDF_AVAILABLE:
        raise RuntimeError("pdfplumber is not installed; cannot handle PDFs.")
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text


# =========================
# Custom CSS & Styling
# =========================

def apply_custom_styling():
    """Apply professional dark theme CSS to the Streamlit app."""
    st.markdown("""
    <style>
    * {
        margin: 0;
        padding: 0;
    }
    
    body, .main {
        background: radial-gradient(circle at top, rgba(59,130,246,0.25), rgba(15,23,42,0.95) 45%),
                    linear-gradient(135deg, #0f1724 0%, #151f2f 45%, #0b1526 100%);
        color: #e0e7ff;
        min-height: 100vh;
        font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        position: relative;
        overflow-x: hidden;
    }
    
    .stApp {
        background: radial-gradient(circle at top, rgba(59,130,246,0.25), rgba(15,23,42,0.95) 45%),
                    linear-gradient(135deg, #0f1724 0%, #151f2f 45%, #0b1526 100%);
        color: #e0e7ff;
    }
    
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top, rgba(59,130,246,0.25), rgba(15,23,42,0.95) 45%),
                    linear-gradient(135deg, #0f1724 0%, #151f2f 45%, #0b1526 100%);
        color: #e0e7ff;
    }
    
    .header-container {
        text-align: center;
        margin-bottom: 40px;
    }
    
    .header-container h1 {
        font-size: clamp(1.8em, 5vw, 3.2em);
        color: #ffffff;
        margin-bottom: 15px;
        font-weight: 700;
        line-height: 1.2;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    
    .header-container p {
        font-size: clamp(1em, 3vw, 1.2em);
        color: #cbd5e1;
        font-weight: 400;
        letter-spacing: 0.5px;
        line-height: 1.6;
    }
    
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border-radius: 9999px;
        border: 1px solid rgba(56, 189, 248, 0.5);
        background: rgba(255, 255, 255, 0.05);
        padding: 6px 16px;
        font-size: 0.9em;
        color: #7dd3fc;
        margin-bottom: 20px;
    }
    
    .badge-dot {
        height: 8px;
        width: 8px;
        border-radius: 50%;
        background-color: #7dd3fc;
        animation: pulse-dot 2.2s ease-in-out infinite;
    }
    
    @keyframes pulse-dot {
        0% { opacity: 0.4; box-shadow: 0 0 0 0 rgba(125, 211, 252, 0.7); }
        50% { opacity: 1; box-shadow: 0 0 0 8px rgba(125, 211, 252, 0); }
        100% { opacity: 0.4; box-shadow: 0 0 0 0 rgba(125, 211, 252, 0); }
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        font-weight: 700;
        border: none;
        font-size: clamp(0.9em, 2vw, 1.2em);
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: clamp(8px, 2vw, 16px) clamp(16px, 3vw, 24px);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4);
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #2c3e50 0%, #34495e 100%);
    }
    
    .prediction-box {
        padding: clamp(20px, 5vw, 40px);
        border-radius: 1.5rem;
        margin-bottom: 30px;
        text-align: center;
        background: linear-gradient(to bottom right, #0f172a, rgba(15, 23, 42, 0.9));
        backdrop-filter: blur(15px);
        color: #e0e7ff;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(148, 163, 184, 0.1);
        animation: fadeIn 0.8s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .confidence-bar-container {
        margin-top: 20px;
    }
    
    .confidence-bar-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.9em;
        color: #94a3b8;
        margin-bottom: 8px;
    }
    
    .confidence-bar-track {
        height: 12px;
        background: rgba(30, 41, 59, 0.8);
        border-radius: 9999px;
        overflow: hidden;
        position: relative;
    }
    
    .confidence-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #0ea5e9, #3b82f6, #6366f1);
        border-radius: 9999px;
        transition: width 1s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .confidence-bar-fill::after {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        animation: shimmer 2s linear infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .info-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 1rem;
        padding: clamp(16px, 4vw, 24px);
        backdrop-filter: blur(16px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    .info-card-title {
        font-size: clamp(0.65em, 1.5vw, 0.75em);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94a3b8;
        margin-bottom: 12px;
    }
    
    .process-step {
        display: flex;
        gap: 12px;
        margin-bottom: 20px;
    }
    
    .step-number {
        flex-shrink: 0;
        width: 38px;
        height: 38px;
        border-radius: 50%;
        background: linear-gradient(135deg, #0ea5e9, #6366f1);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.1em;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3);
    }
    
    .skill-tag {
        display: inline-block;
        background: rgba(59, 130, 246, 0.15);
        padding: 8px 16px;
        border-radius: 25px;
        margin: 5px;
        color: #60a5fa;
        font-size: 0.9em;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(59, 130, 246, 0.3);
        transition: all 0.3s ease;
    }
    
    .skill-tag:hover {
        background: rgba(59, 130, 246, 0.25);
        transform: scale(1.05);
        border-color: rgba(59, 130, 246, 0.5);
    }
    
    /* ===== Tabs ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: rgba(15, 23, 42, 0.7);
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 12px 20px !important;
        font-weight: 600 !important;
        color: #cbd5e1 !important;
        background-color: transparent !important;
        border: none !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
    }
    
    /* ===== Inputs ===== */
    .stTextArea textarea {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        color: #e0e7ff !important;
        border-radius: 10px !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
    }
    
    .stSelectbox, .stMultiSelect {
        background: rgba(30, 41, 59, 0.8) !important;
    }
    
    /* ===== Dataframe ===== */
    .stDataFrame {
        border-radius: 8px !important;
    }
    
    /* ===== Divider ===== */
    .stDivider {
        margin: 2rem 0 !important;
        border-color: rgba(148, 163, 184, 0.1) !important;
    }
    
    /* ===== Expander ===== */
    .stExpander {
        background: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid rgba(148, 163, 184, 0.1) !important;
        border-radius: 8px !important;
    }
    
    /* ===== Responsive Design ===== */
    @media (max-width: 768px) {
        body, .main, .stApp, [data-testid="stAppViewContainer"] {
            background-attachment: fixed;
        }
        
        .header-container {
            margin-bottom: 20px;
            padding: 0 12px;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            padding: 6px;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 8px 12px !important;
            font-size: 0.85em !important;
        }
        
        .prediction-box {
            margin-bottom: 20px;
        }
        
        .skill-tag {
            padding: 6px 12px;
            margin: 4px;
            font-size: 0.8em;
        }
        
        .badge {
            padding: 4px 12px;
            font-size: 0.8em;
        }
    }
    
    @media (max-width: 480px) {
        .header-container {
            margin-bottom: 15px;
            padding: 0 8px;
        }
        
        .header-container h1 {
            margin-bottom: 8px;
        }
        
        .stButton>button {
            height: 3em;
            margin: 8px 0;
        }
        
        .info-card {
            padding: clamp(12px, 3vw, 16px);
        }
        
        .process-step {
            gap: 8px;
            margin-bottom: 12px;
        }
        
        .step-number {
            width: 32px;
            height: 32px;
            font-size: 0.9em;
        }
        
        .prediction-box {
            padding: 15px;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 6px 10px !important;
            font-size: 0.75em !important;
        }
        
        .confidence-bar-container {
            margin-top: 12px;
        }
        
        .skill-tag {
            padding: 4px 8px;
            margin: 2px;
            font-size: 0.75em;
        }
    }
    
    @media (min-width: 769px) and (max-width: 1024px) {
        .header-container h1 {
            font-size: 2.5em;
        }
        
        .header-container p {
            font-size: 1.1em;
        }
    }
    
    </style>
    """, unsafe_allow_html=True)


# =========================
# Streamlit UI
# =========================

def main():
    st.set_page_config(
        page_title="Resume Intelligence Platform",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    apply_custom_styling()

    apply_custom_styling()

    # Header
    st.markdown("""
    <div class="header-container">
        <h1>📄 Resume Intelligence Platform</h1>
        <p>Advanced resume classification and intelligent candidate ranking system</p>
    </div>
    """, unsafe_allow_html=True)

    init_app_state()

    df = st.session_state["candidate_df"]
    
    # Enhanced Stats Section
    st.markdown("""
    <div style="margin-bottom: 30px;">
        <h3 style="color: #ffffff; margin-bottom: 20px; font-size: 1.3em;">Candidate Pool Overview</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    role_emojis = {
        "Software Engineer": "💻",
        "ML Engineer": "🤖",
        "Data Scientist": "📈",
        "Backend / Full-Stack Developer": "⚙️",
        "DevOps / Cloud Engineer": "☁️"
    }
    
    role_counts = df["predicted_role"].value_counts() if len(df) > 0 else {}
    
    # Role Stats Cards - All 5 categories with equal sizing
    cols = [col1, col2, col3, col4, col5]
    
    for i, (role, count) in enumerate(list(role_counts.items())[:5]):
        emoji = role_emojis.get(role, "👤")
        
        with cols[i]:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(125, 211, 252, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%); 
                        border: 2px solid #7dd3fc; border-radius: 12px; padding: 20px 12px; text-align: center; min-height: 160px; display: flex; flex-direction: column; justify-content: center; transition: all 0.3s ease;">
                <div style="font-size: 1.8em; margin-bottom: 12px;">{emoji}</div>
                <div style="font-size: 0.7em; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 12px; font-weight: 600; line-height: 1.2; word-wrap: break-word; word-break: break-word;">{role}</div>
                <div style="font-size: 2.2em; font-weight: 700; color: #7dd3fc;">{count}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(
        ["📄 Classify Resume", "🎯 Rank Candidates", "👥 Candidate Pool"]
    )

    # ----- Tab 1: Classify a resume -----
    with tab1:
        st.markdown("## Classify a Resume")
        
        col_input, col_info = st.columns([2.5, 1])
        
        with col_input:
            mode = st.radio(
                "Input method",
                ["Paste Text", ".txt File", "PDF File"],
                horizontal=True,
                label_visibility="collapsed"
            )

            resume_text = None

            if mode == "Paste Text":
                resume_text = st.text_area(
                    "Resume text",
                    height=280,
                    placeholder="Paste your resume here...",
                    label_visibility="collapsed"
                )

            elif mode == ".txt File":
                uploaded_file = st.file_uploader("Upload .txt file", type=["txt"], label_visibility="collapsed")
                if uploaded_file is not None:
                    resume_text = uploaded_file.read().decode("utf-8", errors="ignore")

            elif mode == "PDF File":
                if not PDF_AVAILABLE:
                    st.error("PDF support not available. Install pdfplumber: pip install pdfplumber")
                else:
                    uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"], label_visibility="collapsed")
                    if uploaded_file is not None:
                        resume_text = pdf_to_text(uploaded_file)
        
        with col_info:
            pass

        if st.button("🔍 Classify", key="btn_classify", use_container_width=True, type="primary"):
            if not resume_text or not resume_text.strip():
                st.error("Please provide resume text")
            else:
                with st.spinner("Analyzing..."):
                    result = classify_resume_text(resume_text)
                    st.session_state["classification_result"] = result
                    st.session_state["classified_resume_text"] = resume_text

        # Display results
        if st.session_state.get("classification_result"):
            result = st.session_state["classification_result"]
            resume_text = st.session_state.get("classified_resume_text", "")

            st.divider()
            st.markdown("### Results")
            
            # Simple results grid
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            
            with res_col1:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Role</div>
                    <div class="result-value">{result['predicted_role']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with res_col2:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Name</div>
                    <div class="result-value">{result['name'] or '—'}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with res_col3:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Email</div>
                    <div class="result-value">{result['email'] or '—'}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with res_col4:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Phone</div>
                    <div class="result-value">{result['phone'] or '—'}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # Action buttons
            col_action1, col_action2 = st.columns([2, 1])
            
            with col_action1:
                if st.button("➕ Add to Pool", key="btn_add_to_pool", use_container_width=True, type="primary"):
                    with st.spinner("Saving..."):
                        add_result = add_resume_text_to_pool(resume_text)
                        
                        if add_result["status"] == "success":
                            st.success(f"✅ Added successfully! ID: {add_result['new_id'][:12]}")
                        elif add_result["status"] == "duplicate":
                            dup = add_result["existing_info"]
                            st.warning(
                                f"**Duplicate Found**\n\n"
                                f"Name: {dup['name']}\n"
                                f"Email: {dup['email']}\n"
                                f"Phone: {dup['phone']}"
                            )
                        else:
                            st.error(f"Error: {add_result['error_msg']}")
            
            with col_action2:
                with st.expander("View Full Text"):
                    st.text_area("", value=resume_text, height=150, disabled=True, label_visibility="collapsed")


    # ----- Tab 2: Rank candidates for JD -----
    with tab2:
        st.markdown("## Rank Candidates for Job Description")

        col_jd, col_settings = st.columns([2, 1])
        
        with col_jd:
            jd_text = st.text_area(
                "Job description",
                height=280,
                placeholder="Paste job description here...",
                label_visibility="collapsed"
            )

        with col_settings:
            st.markdown("**Settings**")
            top_k = st.number_input(
                "Number of candidates",
                min_value=5,
                max_value=30,
                value=10,
                step=1,
                label_visibility="collapsed"
            )

        if st.button("🔍 Rank", key="btn_rank", use_container_width=True, type="primary"):
            if not jd_text or not jd_text.strip():
                st.error("Please paste a job description")
            else:
                with st.spinner("Ranking candidates..."):
                    results_df, jd_info = rank_candidates_for_jd(jd_text, top_k=top_k)

                    st.divider()
                    st.markdown("### Job Analysis")
                    
                    # JD Analysis
                    jd_col1, jd_col2, jd_col3 = st.columns(3)
                    
                    with jd_col1:
                        st.markdown(f"""
                        <div class="result-card">
                            <div class="result-label">Role</div>
                            <div class="result-value">{jd_info['role'] or '—'}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with jd_col2:
                        exp_range = f"{jd_info['min_experience'] or 0}–{jd_info['max_experience'] or 'N/A'} yrs"
                        st.markdown(f"""
                        <div class="result-card">
                            <div class="result-label">Experience</div>
                            <div class="result-value">{exp_range}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with jd_col3:
                        skills_count = len(jd_info['skills'])
                        st.markdown(f"""
                        <div class="result-card">
                            <div class="result-label">Skills Found</div>
                            <div class="result-value">{skills_count}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with st.expander("Detailed analysis"):
                        st.write(f"**Role:** {jd_info['role']}")
                        st.write(f"**Experience:** {jd_info['min_experience']}–{jd_info['max_experience']} years")
                        if jd_info['skills']:
                            st.write(f"**Skills:** {', '.join(sorted(jd_info['skills']))}")

                    st.divider()
                    st.markdown(f"### Top {top_k} Candidates")
                    
                    # Results as professional cards
                    for idx, row in results_df.iterrows():
                        match_pct = f"{row['final_score']:.0%}"
                        
                        st.markdown(f"""
                        <div class="prediction-box" style="border-left: 5px solid #3b82f6;">
                            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr auto; gap: 20px; align-items: center;">
                                <div>
                                    <div style="font-size: 0.85em; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;">Name</div>
                                    <div style="font-size: 1.1em; font-weight: 600; color: #ffffff;">{row['name']}</div>
                                </div>
                                <div>
                                    <div style="font-size: 0.85em; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;">Contact</div>
                                    <div style="font-size: 0.95em; color: #cbd5e1;">{row['email']}<br>{row['phone']}</div>
                                </div>
                                <div>
                                    <div style="font-size: 0.85em; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;">Role</div>
                                    <div style="font-size: 1em; color: #7dd3fc; font-weight: 600;">{row['predicted_role']}</div>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-size: 0.85em; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;">Match</div>
                                    <div style="font-size: 2em; font-weight: 700; color: #3b82f6;">{match_pct}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.divider()
                    
                    # Download
                    dl_col1, dl_col2 = st.columns(2)
                    
                    with dl_col1:
                        csv_data = results_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "📥 Download CSV",
                            data=csv_data,
                            file_name=f"ranked_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with dl_col2:
                        json_data = results_df.to_json(orient='records', indent=2)
                        st.download_button(
                            "📥 Download JSON",
                            data=json_data,
                            file_name=f"ranked_{pd.Timestamp.now().strftime('%Y%m%d')}.json",
                            mime="application/json",
                            use_container_width=True
                        )


    # ----- Tab 3: Candidate pool viewer -----
    with tab3:
        st.markdown("## Candidate Pool")

        df = st.session_state["candidate_df"]
        
        # Filters
        st.markdown("**Filters**")
        
        f_col1, f_col2 = st.columns([1, 2])

        with f_col1:
            roles = sorted(df["predicted_role"].dropna().unique().tolist())
            selected_roles = st.multiselect(
                "Filter by role",
                roles,
                default=roles,
                label_visibility="collapsed"
            )

        with f_col2:
            query = st.text_input(
                "Search",
                value="",
                placeholder="Search name or email...",
                label_visibility="collapsed"
            )

        st.divider()
        
        # Filter data
        filtered = df[df["predicted_role"].isin(selected_roles)]

        if query:
            q = query.lower()
            filtered = filtered[
                filtered["name"].fillna("").str.lower().str.contains(q, na=False)
                | filtered["email"].fillna("").str.lower().str.contains(q, na=False)
            ]

        st.markdown(f"**Showing {len(filtered)} candidates** (total: {len(df)})")
        
        # Display table
        show_cols = ["id", "name", "email", "phone", "predicted_role", "experience_years_num"]
        existing_cols = [c for c in show_cols if c in filtered.columns]

        display_df = filtered[existing_cols].copy()
        display_df.columns = ['ID', 'Name', 'Email', 'Phone', 'Role', 'Exp (yrs)']

        st.dataframe(display_df, use_container_width=True, hide_index=True, height=450)

        st.divider()

        # Download options
        st.markdown("**Export**")
        d_col1, d_col2, d_col3 = st.columns(3)
        
        with d_col1:
            csv_filtered = filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Filtered (CSV)",
                data=csv_filtered,
                file_name=f"candidates_filtered_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with d_col2:
            csv_all = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 All (CSV)",
                data=csv_all,
                file_name=f"candidates_all_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with d_col3:
            json_data = filtered.to_json(orient='records', indent=2)
            st.download_button(
                "📥 Filtered (JSON)",
                data=json_data,
                file_name=f"candidates_{pd.Timestamp.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )


if __name__ == "__main__":
    main()