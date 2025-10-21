# ═══════════════════════════════════════════════════════════════════
# STREAMLIT APP - VERIDIA.IO RESUME CLASSIFIER (ENHANCED VERSION)
# ═══════════════════════════════════════════════════════════════════

import streamlit as st
import torch
import pdfplumber
import docx
import re
import pickle
from transformers import BertTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import heapq

# ═══════════════════════════════════════════════════════════════════
# PATH CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
MODEL_PATH = "/content/drive/MyDrive/resume_classifier_best"
LABEL_ENCODER_PATH = "/content/label_encoder.pkl"

# ═══════════════════════════════════════════════════════════════════
# TEXT CLEANING FUNCTION
# (SAME AS USED IN TRAINING)
# ═══════════════════════════════════════════════════════════════════
def clean_resume_text(text):
    text = str(text) if text else ""
    text = re.sub(r'http\S+|www\.\S+|\S+@\S+|<.*?>|#\S+|@\S+|\bRT\b|\bcc\b', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

# ═══════════════════════════════════════════════════════════════════
# LOAD MODEL AND TOKENIZER
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, label_encoder, device

model, tokenizer, label_encoder, device = load_model()

# ═══════════════════════════════════════════════════════════════════
# PDF / DOCX EXTRACTOR
# ═══════════════════════════════════════════════════════════════════
def extract_text_from_file(uploaded_file):
    text = ""
    if uploaded_file.name.endswith(".pdf"):
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    elif uploaded_file.name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        text = uploaded_file.read().decode("utf-8", errors="ignore")
    return text

# ═══════════════════════════════════════════════════════════════════
# PREDICTION FUNCTION (TOP-5 OUTPUT)
# ═══════════════════════════════════════════════════════════════════
def predict_resume_category(text):
    cleaned = clean_resume_text(text)
    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    # Get top-5 predictions
    top5_idx = np.argsort(probs)[::-1][:5]
    top5_labels = label_encoder.inverse_transform(top5_idx)
    top5_scores = probs[top5_idx]
    top5 = list(zip(top5_labels, top5_scores))

    final_label = top5_labels[0]
    final_conf = top5_scores[0]

    return final_label, final_conf, top5, cleaned

# ═══════════════════════════════════════════════════════════════════
# KEY PHRASE EXTRACTION (SIMPLE HIGHLIGHT BASED ON CATEGORY)
# ═══════════════════════════════════════════════════════════════════
def extract_key_phrases(text, predicted_label):
    """
    Simple rule-based keyword importance:
    It extracts lines containing keywords related to the predicted label.
    """
    label_keywords = {
        "python developer": ["python", "django", "flask", "pandas", "numpy", "api", "machine learning"],
        "data scientist": ["data", "model", "machine learning", "analysis", "ai", "statistics", "prediction"],
        "web developer": ["html", "css", "javascript", "react", "node", "frontend", "backend", "web"],
        "android developer": ["android", "java", "kotlin", "mobile", "app", "firebase", "xml"],
        "network engineer": ["network", "router", "switch", "tcp", "ip", "firewall", "vpn"],
        "software engineer": ["software", "development", "testing", "design", "debugging", "programming"],
        "devops engineer": ["docker", "kubernetes", "ci/cd", "aws", "jenkins", "automation"],
        "database administrator": ["database", "sql", "mysql", "oracle", "postgresql", "nosql", "query"],
    }

    phrases = []
    lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 10]
    keywords = label_keywords.get(predicted_label.lower(), [])

    for line in lines:
        for key in keywords:
            if key.lower() in line.lower():
                phrases.append(line)
                break

    return phrases[:5] if phrases else ["No specific key phrases found that strongly influenced the prediction."]

# ═══════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Veridia.io Resume Classifier", page_icon="🤖", layout="wide")

st.title("🤖 Veridia.io - AI Resume Classification System")
st.write("Upload a resume (PDF/DOCX/TXT) or paste the text below to classify it into its professional category with confidence insights.")

uploaded_file = st.file_uploader("📁 Upload Resume File", type=["pdf", "docx", "txt"])
input_text = st.text_area("📝 Or Paste Resume Text Here", height=200)

if st.button("🔍 Classify Resume"):
    if uploaded_file is not None:
        text = extract_text_from_file(uploaded_file)
    else:
        text = input_text.strip()

    if not text:
        st.warning("⚠️ Please upload a file or paste text to classify.")
    else:
        with st.spinner("Analyzing resume..."):
            final_label, final_conf, top5, cleaned_text = predict_resume_category(text)
            key_phrases = extract_key_phrases(cleaned_text, final_label)

        # Display Results
        st.markdown(f"## 🏆 Final Predicted Category: **{final_label}**")
        st.markdown(f"**Confidence:** {final_conf*100:.2f}%")

        st.markdown("---")
        st.subheader("📊 Top 5 Category Probabilities")

        df_top5 = pd.DataFrame(top5, columns=["Category", "Confidence"])
        df_top5["Confidence (%)"] = (df_top5["Confidence"] * 100).round(2)
        st.bar_chart(df_top5.set_index("Category")["Confidence (%)"])

        st.markdown("---")
        st.subheader("🧩 Key Lines Influencing Prediction")
        for phrase in key_phrases:
            st.markdown(f"- {phrase}")

st.markdown("---")
st.caption("Developed by Veridia.io | Powered by BERT and Streamlit 🚀")
