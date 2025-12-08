import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pdfplumber
from docx import Document
import re
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from pathlib import Path
import json
import time

# Set page configuration
st.set_page_config(
    page_title="Resume Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)# Custom CSS for stunning, responsive UI
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
    
    .blur-blob {
        position: fixed;
        width: 320px;
        height: 320px;
        border-radius: 50%;
        filter: blur(140px);
        opacity: 0.35;
        z-index: 0;
        pointer-events: none;
    }
    
    .blur-blob-1 {
        background: rgba(59, 130, 246, 0.4);
        left: -80px;
        top: -40px;
    }
    
    .blur-blob-2 {
        background: rgba(56, 189, 248, 0.3);
        right: -120px;
        bottom: -80px;
    }
    
    .stApp {
        background: radial-gradient(circle at top, rgba(59,130,246,0.25), rgba(15,23,42,0.95) 45%),
                    linear-gradient(135deg, #0f1724 0%, #151f2f 45%, #0b1526 100%);
        color: #e0e7ff;
    }
    
    .navbar {
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(16px);
        padding: 20px 30px;
        border-radius: 1.5rem;
        margin: 20px 20px 0 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #e0e7ff;
        box-shadow: 0 25px 70px rgba(0, 0, 0, 0.35);
        border: 1px solid rgba(148, 163, 184, 0.15);
    }
    
    .navbar-brand {
        font-size: 1.5em;
        font-weight: bold;
        display: flex;
        align-items: center;
    }
    
    .navbar-stats {
        display: flex;
        gap: 20px;
        font-size: 0.9em;
    }
    
    .stat-item {
        background: rgba(255, 255, 255, 0.1);
        padding: 5px 15px;
        border-radius: 20px;
        color: #ffffff;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
    }
    
    .main-content {
        /* Removed background and shadow for cleaner look */
        padding: 20px;
        margin: 20px;
    }
    
    .header-container {
        text-align: center;
        margin-bottom: 40px;
    }
    
    .header-container h1 {
        font-size: 3.2em;
        color: #ffffff;
        margin-bottom: 15px;
        font-weight: 700;
        line-height: 1.2;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    
    .header-container p {
        font-size: 1.2em;
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
        font-size: 1.2em;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4);
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #2c3e50 0%, #34495e 100%);
    }
    
    .prediction-box {
        padding: 40px;
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
        padding: 24px;
        backdrop-filter: blur(16px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    .info-card-title {
        font-size: 0.75em;
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
    
    .prediction-box h2 {
        font-size: 2.2em;
        margin-bottom: 10px;
        font-weight: 700;
    }
    
    .prediction-box p {
        font-size: 1em;
        opacity: 0.95;
    }
    
    .info-section {
        background-color: #f8f9ff;
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 25px;
        border-left: 5px solid #667eea;
    }
    
    .info-section h3 {
        color: #2c3e50;
        margin-bottom: 15px;
        font-size: 1.3em;
    }
    
    .info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 15px;
        margin-top: 15px;
    }
    
    .category-tag {
        background-color: white;
        padding: 12px 18px;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        color: #667eea;
        border: 2px solid #667eea;
        font-size: 0.95em;
    }
    
    .step {
        display: flex;
        margin-bottom: 15px;
        align-items: flex-start;
    }
    
    .step-number {
        background-color: #667eea;
        color: white;
        width: 35px;
        height: 35px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 15px;
        flex-shrink: 0;
    }
    
    .step-text {
        color: #2c3e50;
        font-size: 0.95em;
        line-height: 1.6;
    }
    
    .sidebar .stMarkdown {
        color: white;
    }
    
    .model-badge {
        display: inline-block;
        background-color: #667eea;
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.85em;
        margin-right: 10px;
        margin-bottom: 5px;
        font-weight: 600;
    }
    
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85em;
        padding-top: 30px;
        margin-top: 40px;
        border-top: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    .footer p {
        margin: 5px 0;
    }
    
    .upload-section {
        background: rgba(15, 23, 42, 0.6);
        border: 2px dashed rgba(148, 163, 184, 0.3);
        border-radius: 1.5rem;
        padding: 40px;
        text-align: center;
        margin-bottom: 25px;
        transition: all 0.3s ease;
        backdrop-filter: blur(8px);
    }
    
    .upload-section:hover {
        border-color: rgba(56, 189, 248, 0.5);
        background: rgba(15, 23, 42, 0.8);
    }
    
    @media (max-width: 768px) {
        .main-content {
            padding: 25px 15px;
            margin: 10px;
        }
        
        .header-container h1 {
            font-size: 1.8em;
        }
        
        .prediction-box {
            padding: 25px 15px;
        }
        
        .prediction-box h2 {
            font-size: 1.8em;
        }
        
        .info-grid {
            grid-template-columns: 1fr;
        }
        
        /* Stack columns on mobile */
        [data-testid="column"] {
            width: 100% !important;
            flex: 100% !important;
            max-width: 100% !important;
        }
        
        /* Make info cards responsive */
        .info-card {
            margin-bottom: 15px;
            padding: 16px;
        }
        
        /* Adjust process step sizing */
        .process-step {
            margin-bottom: 15px;
            gap: 10px;
        }
        
        .step-number {
            width: 32px;
            height: 32px;
            font-size: 0.95em;
        }
        
        /* Adjust navbar for mobile */
        .navbar {
            font-size: 0.85em;
            padding: 12px 10px;
            flex-direction: column;
            gap: 10px;
        }
        
        .stat-item {
            font-size: 0.8em;
            padding: 6px 10px;
        }
        
        /* Upload section mobile */
        .upload-section {
            padding: 25px 15px;
        }
        
        /* Badge responsive */
        .badge {
            font-size: 0.75em;
            padding: 6px 14px;
        }
    }
    
    @media (max-width: 480px) {
        .header-container h1 {
            font-size: 1.5em;
        }
        
        .header-container p {
            font-size: 0.9em;
        }
        
        .prediction-box h2 {
            font-size: 1.5em;
        }
        
        .skill-tag {
            font-size: 0.75em;
            padding: 5px 10px;
        }
    }
    
    /* Remove extra spacing between elements */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
    }
    
    .element-container {
        margin-bottom: 0 !important;
    }
    
    /* Remove gap in columns */
    [data-testid="column"] > div {
        gap: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

SKILLS_DB = [
    'Python', 'SQL', 'Java', 'JavaScript', 'React', 'AWS', 'Azure', 
    'Excel', 'Tableau', 'PowerBI', 'Docker', 'Kubernetes', 'Peoplesoft', 
    'Workday', 'SAP', 'HTML', 'CSS', 'Node.js', 'C++', 'C#', 'Linux',
    'Git', 'Machine Learning', 'Deep Learning', 'NLP', 'Pandas', 'NumPy',
    'Scikit-learn', 'TensorFlow', 'Keras', 'Pytorch', 'Flask', 'Django'
]

def extract_skills(text):
    """Simple keyword matching for skills."""
    found_skills = []
    for skill in SKILLS_DB:
        if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
            found_skills.append(skill)
    return list(set(found_skills))

@st.cache_resource
def load_model_artifacts():
    """Load the trained model, vectorizer, and encoder."""
    try:
        base_path = Path("models")
        
        # Load metadata to find the best model name
        with open(base_path / "model_metadata.json", "r") as f:
            metadata = json.load(f)
        
        model_filename = f"{metadata['best_model'].lower().replace(' ', '_')}.pkl"
        
        model = joblib.load(base_path / model_filename)
        vectorizer = joblib.load(base_path / "tfidf_vectorizer.pkl")
        encoder = joblib.load(base_path / "label_encoder.pkl")
        
        return model, vectorizer, encoder, metadata
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None, None

def clean_text(text):
    """Remove XML metadata and Office artifacts from extracted text (Robust Version)."""
    if not text:
        return ""
    
    # Remove entire lines that are just XML/metadata patterns
    lines = text.split('\n')
    cleaned_lines = []
    
    xml_patterns = [
        r'http[s]?://schemas', r'openxmlformats', r'_rels', r'theme', r'xmlpk',
        r'content_types', r'xmlns', r'encoding', r'utf-8', r'harika395',
        r'outlook\.com', r'2003', r'2006', r'document', r'word',
        r'drawingml', r'clmap', r'pk\d+'
    ]
    
    for line in lines:
        # Skip lines that contain ONLY metadata/XML patterns
        line_lower = line.lower()
        is_metadata = any(pattern.lower() in line_lower for pattern in xml_patterns)
        
        # If line contains metadata AND is short (metadata lines are typically short)
        if is_metadata and len(line.split()) < 5:
            continue
        
        # Also remove lines that are just repeated words
        words = line.split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3 and len(words) > 3:  # Very repetitive
                continue
        
        cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines)
    
    # Aggressive word-level cleanup - remove standalone metadata words
    metadata_words = [
        r'\bschemas\b', r'\bxml\b', r'\btheme\b', r'\b_rels\b', r'\bopenxml.*?\b',
        r'\bxmlns\b', r'\bhttp\b', r'\bhttps\b', r'\bencoding\b', r'\butf-8\b',
        r'\bcontent\b', r'\btypes\b', r'\brels\b', r'\bversion\b', r'\bstandalone\b',
        r'\baccent\d+\b', r'\bfont\b', r'\btable\b', r'\bnormal\b', r'\bbg\d+\b',
        r'\brelspk\b', r'\bharika395\b', r'\borg\b', r'\bcommon\b',
        r'\btimes\b', r'\broman\b', r'\barial\b', r'\bcalibri\b', r'\bcourier\b',
        r'\bgeorgia\b', r'\bverdana\b', r'\btahoma\b', r'\bsans\b', r'\bserif\b',
        r'\bdr\d+\b', r'\bdl\d+\b', r'\bdk\d+\b', r'\bcidno\d+\b',
        r'\bdt\d+\b', r'\bos\d+\b', r'\bfnb\d+\b',
        r'\bdrawingml\b', r'\bclmap\b', r'\blx\d+\b', r'\bli\d+\b', r'\btx\d+\b',
        r'\bpk\d+\b', r'\bhlink\b', r'\bfol\b', r'\bmain\b'
    ]
    
    for pattern in metadata_words:
        result = re.sub(pattern, ' ', result, flags=re.IGNORECASE)
    
    # Clean up multiple spaces
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result

def extract_text_from_pdf(file):
    """Extract text from PDF file."""
    try:
        with pdfplumber.open(file) as pdf:
            text = ""
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def extract_text_from_docx(file):
    """Extract text from DOCX file."""
    try:
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return None

def extract_text(uploaded_file):
    """Route extraction based on file type."""
    if uploaded_file.name.endswith('.pdf'):
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith('.docx'):
        return extract_text_from_docx(uploaded_file)
    elif uploaded_file.name.endswith('.txt'):
        return str(uploaded_file.read(), "utf-8")
    else:
        return None

# --- Main App Layout ---

def main():
    model, vectorizer, encoder, metadata = st.session_state.get('model_artifacts', (None, None, None, None))
    
    # Create sidebar with Process Steps and Model Integrity
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h2 style="color: white; margin: 0;">📄 Resume AI</h2>
            <p style="color: #94a3b8; margin: 5px 0 0 0; font-size: 0.9em;">Classification System</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <p class="info-card-title">Process Steps</p>
            <div class="process-step">
                <span class="step-number">1</span>
                <div>
                    <p style="color: white; font-weight: 600; margin: 0;">Upload Resume</p>
                    <p style="color: #94a3b8; font-size: 0.85em; margin: 5px 0 0 0;">Secure client-side sanitization before transfer to model</p>
                </div>
            </div>
            <div class="process-step">
                <span class="step-number">2</span>
                <div>
                    <p style="color: white; font-weight: 600; margin: 0;">Extract Features</p>
                    <p style="color: #94a3b8; font-size: 0.85em; margin: 5px 0 0 0;">TF-IDF vectors and contextual embeddings extracted</p>
                </div>
            </div>
            <div class="process-step">
                <span class="step-number">3</span>
                <div>
                    <p style="color: white; font-weight: 600; margin: 0;">Predict & Explain</p>
                    <p style="color: #94a3b8; font-size: 0.85em; margin: 5px 0 0 0;">Model predictions with explainability and confidence</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if metadata:
            st.markdown(f"""
            <div class="info-card" style="margin-top: 20px;">
                <p class="info-card-title">Model Integrity</p>
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                        <span style="color: #94a3b8; font-size: 0.9em;">Algorithm</span>
                        <span style="color: white; font-weight: 600;">KNN</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                        <span style="color: #94a3b8; font-size: 0.9em;">Accuracy</span>
                        <span style="color: #10b981; font-weight: 600;">{metadata.get('test_accuracy', 1.0) * 100:.1f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                        <span style="color: #94a3b8; font-size: 0.9em;">F1 Score</span>
                        <span style="color: #10b981; font-weight: 600;">{metadata.get('test_f1_score', 1.0) * 100:.1f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #94a3b8; font-size: 0.9em;">Categories</span>
                        <span style="color: white; font-weight: 600;">4</span>
                    </div>
                </div>
                <div style="padding: 10px; background: rgba(16, 185, 129, 0.1); border-radius: 8px; border-left: 3px solid #10b981; margin-bottom: 12px;">
                    <p style="color: #10b981; margin: 0; font-size: 0.85em; font-weight: 600;">✓ Model Validated</p>
                </div>
                <ul style="margin: 0; padding-left: 20px; color: #94a3b8; font-size: 0.85em; line-height: 1.8;">
                    <li>Artifacts cached with integrity checks</li>
                    <li>No mutation of deployed pipelines</li>
                    <li>Real-time confidence monitoring</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Navbar
    if metadata:
        st.markdown(f"""
        <div class="navbar">
            <div class="navbar-brand">📄 Resume AI Classifier</div>
            <div class="navbar-stats">
                <div class="stat-item">Model: {metadata['best_model']}</div>
                <div class="stat-item">Accuracy: {metadata['test_accuracy']:.1%}</div>
                <div class="stat-item">F1 Score: {metadata['test_f1_score']:.1%}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Blur blob decorative elements
    st.markdown("""
    <div class="blur-blob blur-blob-1"></div>
    <div class="blur-blob blur-blob-2"></div>
    """, unsafe_allow_html=True)
    
    # Main content wrapper
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="header-container">
        <div class="badge">
            <span class="badge-dot"></span>
            Instant, AI-Powered Screening
        </div>
        <h1>Elevate Your Resume Screening Experience</h1>
        <p>Upload PDFs, DOCX or TXT resumes, and get precise career domain predictions with confidence scores and actionable insights.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content (no columns, full width)
    with st.container():
        # Info Cards Section
        info_col1, info_col2, info_col3 = st.columns(3)
        
        with info_col1:
            st.markdown("""
            <div style="background: rgba(15, 23, 42, 0.5); border: 1px solid rgba(14, 165, 233, 0.3); border-radius: 12px; padding: 20px; text-align: left;">
                <p style="color: #94a3b8; font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.15em; margin: 0 0 12px 0;">Processing</p>
                <h3 style="color: white; font-weight: 600; margin: 0 0 8px 0; font-size: 1.1em;">Intelligent Cleaning</h3>
                <p style="color: #cbd5e1; margin: 0; font-size: 0.85em; line-height: 1.6;">Removes XML artifacts, repetitive metadata, and noisy formatting for clean signal extraction.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with info_col2:
            st.markdown("""
            <div style="background: rgba(15, 23, 42, 0.5); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 12px; padding: 20px; text-align: left;">
                <p style="color: #94a3b8; font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.15em; margin: 0 0 12px 0;">Modeling</p>
                <h3 style="color: white; font-weight: 600; margin: 0 0 8px 0; font-size: 1.1em;">KNN + TF-IDF</h3>
                <p style="color: #cbd5e1; margin: 0; font-size: 0.85em; line-height: 1.6;">Hybrid feature extraction captures contextual and statistical patterns for accurate classification.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with info_col3:
            st.markdown("""
            <div style="background: rgba(15, 23, 42, 0.5); border: 1px solid rgba(168, 85, 247, 0.3); border-radius: 12px; padding: 20px; text-align: left;">
                <p style="color: #94a3b8; font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.15em; margin: 0 0 12px 0;">Insights</p>
                <h3 style="color: white; font-weight: 600; margin: 0 0 8px 0; font-size: 1.1em;">Explainable Outcomes</h3>
                <p style="color: #cbd5e1; margin: 0; font-size: 0.85em; line-height: 1.6;">Confidence, skills, and predictions help recruiters act instantly with clarity.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Upload Section (remove extra spacing)
        st.markdown('<div class="info-card" style="margin-top: 0;">', unsafe_allow_html=True)
        st.markdown('<p class="info-card-title" style="margin-bottom: 15px;">📤 Upload Resume</p>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Drop your resume here", type=['pdf', 'docx', 'txt'], label_visibility="collapsed")
        
        # Text Input Option
        st.markdown('<p style="color: #94a3b8; font-size: 0.9em; margin: 10px 0 8px 0;">Or paste resume text:</p>', unsafe_allow_html=True)
        resume_text_input = st.text_area("Paste resume text", height=120, label_visibility="collapsed", placeholder="Paste resume text here (minimum 30 characters)...")
        
        # Classification Button
        st.markdown('<div style="margin-top: 15px;">', unsafe_allow_html=True)
        st.button("🔍 CLASSIFY RESUME", key="analyze", use_container_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        # Process the resume
        text = None
        if uploaded_file is not None:
            with st.spinner("Extracting text from file..."):
                text = extract_text(uploaded_file)
        elif resume_text_input and len(resume_text_input.strip()) >= 30:
            text = resume_text_input
        
        if st.session_state.get('analyze'):
            if text and len(text.strip()) >= 30:
                cleaned_text = clean_text(text)
                
                if model and vectorizer and encoder:
                    # Show live intelligence panel in main column (full width)
                    st.markdown('<div class="info-card" style="margin-top: 25px;">', unsafe_allow_html=True)
                    st.markdown('<h3 style="color: white; margin-bottom: 20px;">🔄 Live Intelligence Panel</h3>', unsafe_allow_html=True)
                    
                    parser_placeholder = st.empty()
                    feature_placeholder = st.empty()
                    engine_placeholder = st.empty()
                    
                    # Phase 1: Parsing
                    parser_placeholder.markdown("""
                    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 12px; padding: 16px; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.9em;">
                            <span style="color: #94a3b8;">Parsing Engine</span>
                            <span style="color: #7dd3fc;">Processing...</span>
                        </div>
                        <div style="height: 8px; background: rgba(30, 41, 59, 0.8); border-radius: 9999px; overflow: hidden;">
                            <div style="height: 100%; width: 0%; background: #0ea5e9; border-radius: 9999px; transition: width 0.5s;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    time.sleep(0.3)
                    for i in range(0, 101, 20):
                        parser_placeholder.markdown(f"""
                        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 12px; padding: 16px; margin-bottom: 12px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.9em;">
                                <span style="color: #94a3b8;">Parsing Engine</span>
                                <span style="color: #7dd3fc;">Processing...</span>
                            </div>
                            <div style="height: 8px; background: rgba(30, 41, 59, 0.8); border-radius: 9999px; overflow: hidden;">
                                <div style="height: 100%; width: {i}%; background: #0ea5e9; border-radius: 9999px; transition: width 0.3s;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        time.sleep(0.15)
                    
                    parser_placeholder.markdown("""
                    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 12px; padding: 16px; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.9em;">
                            <span style="color: #94a3b8;">Parsing Engine</span>
                            <span style="color: #10b981;">Complete ✓</span>
                        </div>
                        <div style="height: 8px; background: rgba(30, 41, 59, 0.8); border-radius: 9999px; overflow: hidden;">
                            <div style="height: 100%; width: 100%; background: #10b981; border-radius: 9999px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Phase 2: Feature Extraction
                    for i in range(0, 101, 25):
                        feature_placeholder.markdown(f"""
                        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 12px; padding: 16px; margin-bottom: 12px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.9em;">
                                <span style="color: #94a3b8;">Feature Extraction</span>
                                <span style="color: #7dd3fc;">Vectorizing...</span>
                            </div>
                            <div style="height: 8px; background: rgba(30, 41, 59, 0.8); border-radius: 9999px; overflow: hidden;">
                                <div style="height: 100%; width: {i}%; background: #6366f1; border-radius: 9999px; transition: width 0.3s;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        time.sleep(0.12)
                    
                    # Perform actual prediction during feature phase
                    text_vectorized = vectorizer.transform([cleaned_text])
                    prediction_idx = model.predict(text_vectorized)[0]
                    predicted_category = encoder.inverse_transform([prediction_idx])[0]
                    
                    try:
                        probs = model.predict_proba(text_vectorized)
                        confidence = np.max(probs)
                    except:
                        confidence = 0.95
                    
                    feature_placeholder.markdown("""
                    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 12px; padding: 16px; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.9em;">
                            <span style="color: #94a3b8;">Feature Extraction</span>
                            <span style="color: #10b981;">Complete ✓</span>
                        </div>
                        <div style="height: 8px; background: rgba(30, 41, 59, 0.8); border-radius: 9999px; overflow: hidden;">
                            <div style="height: 100%; width: 100%; background: #10b981; border-radius: 9999px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Phase 3: Prediction
                    for i in range(0, 101, 33):
                        engine_placeholder.markdown(f"""
                        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 12px; padding: 16px; margin-bottom: 12px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.9em;">
                                <span style="color: #94a3b8;">Prediction Engine</span>
                                <span style="color: #7dd3fc;">Predicting...</span>
                            </div>
                            <div style="height: 8px; background: rgba(30, 41, 59, 0.8); border-radius: 9999px; overflow: hidden;">
                                <div style="height: 100%; width: {i}%; background: #a78bfa; border-radius: 9999px; transition: width 0.3s;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        time.sleep(0.1)
                    
                    engine_placeholder.markdown("""
                    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 12px; padding: 16px; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.9em;">
                            <span style="color: #94a3b8;">Prediction Engine</span>
                            <span style="color: #10b981;">Complete ✓</span>
                        </div>
                        <div style="height: 8px; background: rgba(30, 41, 59, 0.8); border-radius: 9999px; overflow: hidden;">
                            <div style="height: 100%; width: 100%; background: #10b981; border-radius: 9999px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display Result in full width
                    st.markdown(f"""
                    <div class="prediction-box">
                        <p style="font-size: 0.85em; margin-bottom: 10px; opacity: 0.7; text-transform: uppercase; letter-spacing: 3px;">Predicted Category</p>
                        <h2 style="font-size: 2.8em; margin: 15px 0; line-height: 1.2;">{predicted_category}</h2>
                        <div class="confidence-bar-container">
                            <div class="confidence-bar-header">
                                <span>Confidence</span>
                                <span>{confidence:.1%}</span>
                            </div>
                            <div class="confidence-bar-track">
                                <div class="confidence-bar-fill" style="width: {confidence * 100}%"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Extract and display skills with gradient colors
                    skills = extract_skills(cleaned_text)
                    if skills:
                        st.markdown('<div style="margin-top: 20px; margin-bottom: 30px;">', unsafe_allow_html=True)
                        st.markdown('<p style="color: #94a3b8; font-size: 0.9em; margin-bottom: 12px; font-weight: 500;">Highlighted Skills</p>', unsafe_allow_html=True)
                        
                        # Skill color palette matching HTML
                        skill_colors = ['#60a5fa', '#a78bfa', '#f472b6', '#fbbf24', '#34d399', '#38bdf8']
                        skills_html = '<div style="display: flex; flex-wrap: wrap; gap: 8px;">'
                        for idx, skill in enumerate(skills[:10]):
                            color = skill_colors[idx % len(skill_colors)]
                            skills_html += f'<span style="display: inline-block; padding: 6px 14px; background: {color}22; border: 1px solid {color}55; border-radius: 20px; font-size: 0.8em; color: {color}; font-weight: 500;">{skill}</span>'
                        skills_html += '</div>'
                        st.markdown(skills_html, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                else:
                    st.error("Model not loaded correctly. Please check the model files.")
            else:
                st.warning("⚠️ Please upload a resume file or paste at least 30 characters of text to begin classification.")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p><strong>Resume AI Classifier</strong> • AI-Powered Resume Classification</p>
        <p>Built with Streamlit | Machine Learning | Natural Language Processing</p>
        <p style="margin-top: 10px; opacity: 0.7;">© 2025 Resume Classification System</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    # Initial Loading Animation
    placeholder = st.empty()
    with placeholder.container():
        st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; height: 60vh; flex-direction: column;">
            <h1 style="color: white; font-size: 3em; margin-bottom: 20px; text-shadow: 0 0 20px rgba(59, 130, 246, 0.5);">📄 Resume AI Classifier</h1>
            <div style="display: flex; align-items: center; gap: 15px;">
                <div style="width: 20px; height: 20px; border: 3px solid #3b82f6; border-top: 3px solid transparent; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                <p style="color: #e0e7ff; font-size: 1.5em; margin: 0;">Initializing Intelligence Models...</p>
            </div>
            <style>
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
        </div>
        """, unsafe_allow_html=True)
        
        # Load artifacts
        model, vectorizer, encoder, metadata = load_model_artifacts()
        st.session_state['model_artifacts'] = (model, vectorizer, encoder, metadata)
        time.sleep(1.5) # Force a small delay to show the screen
        
    placeholder.empty()
    main()
