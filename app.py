"""
Sentiment Analysis & Topic Modeling Research Application
=========================================================
Main Streamlit entry point.

Tabs:
  0. Data Pipeline & Overview   - Load, clean, inspect comments
  A. Topic Modeling             - Multi-algorithm construct classification
  B. Metrics & Comparative      - Evaluation, keyword analysis, brand comparison
  C. Sentiment-Topic Integration - Emotion x topic joint analysis

Usage:
    streamlit run app.py
"""

import streamlit as st
import nltk
import warnings
warnings.filterwarnings("ignore")

# Ensure NLTK data is available
for resource in ["stopwords", "punkt", "punkt_tab", "wordnet", "vader_lexicon",
                  "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}" if resource in ("stopwords", "wordnet") else f"sentiment/{resource}" if "vader" in resource else f"taggers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

# --- Page config ---
st.set_page_config(
    page_title="YouTube Comment Analysis - Sentiment & Topic Modeling",
    page_icon="research",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar ---
st.sidebar.title("Research Pipeline")
st.sidebar.markdown(
    "**YouTube Comment Analysis**\n\n"
    "Sentiment Analysis & Topic Modeling across "
    "Xiaomi, Samsung, and Huawei."
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Comments_and_brands.xlsx", type=["xlsx", "xls"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Constructs:** Trust, Perceived Quality, Entertainment, "
    "Usefulness, Price Value Perception, eWOM/Recommendation, "
    "Product Information"
)
st.sidebar.markdown(
    "**Emotions:** sadness, joy, love, anger, fear, surprise"
)
st.sidebar.markdown(
    "**Algorithms:** Keyword, LDA, NMF, Seeded LDA, BERTopic, Zero-Shot"
)

# --- Tabs ---
st.title("YouTube Comment Analysis Pipeline")
st.markdown("*Sentiment Analysis & Topic Modeling for Consumer Behavior Research*")

tab0, tab_a, tab_b, tab_c = st.tabs([
    "Data Pipeline",
    "Topic Modeling",
    "Metrics & Analysis",
    "Sentiment-Topic Integration",
])

# Import and render each tab
from tabs.tab_overview import render_tab as render_overview
from tabs.tab_topic_modeling import render_tab as render_topic_modeling
from tabs.tab_metrics import render_tab as render_metrics
from tabs.tab_sentiment_topics import render_tab as render_sentiment_topics

with tab0:
    render_overview(uploaded_file=uploaded_file)

with tab_a:
    render_topic_modeling()

with tab_b:
    render_metrics()

with tab_c:
    render_sentiment_topics()
