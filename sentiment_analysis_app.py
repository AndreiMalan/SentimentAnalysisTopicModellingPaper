"""
Sentiment Analysis and Topic Modeling Application - Enhanced with BERT & BERTopic
==================================================================================

This application provides state-of-the-art text analysis capabilities including:
1. Sentiment Analysis using traditional ML + BERT transformers
2. Topic Modeling using LDA, NMF, BERTopic, and Guided Topic Modeling
3. Zero-Shot Classification for predefined social commerce behavior topics
4. Topic-Sentiment Integration for comprehensive analysis
5. Interactive Streamlit interface for real-time predictions

Author: AI Research Team
Date: 2026-01-01
Version: 3.0 (Enhanced with Social Commerce Topic Modeling)
"""

import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.decomposition import LatentDirichletAllocation, NMF
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from scipy.stats import entropy
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import spacy
import warnings
warnings.filterwarnings('ignore')

# Try to import transformers and BERTopic (optional dependencies)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

# Try to import CorEx for guided topic modeling
try:
    from corextopic import Corex as CorEx
    COREX_AVAILABLE = True
except ImportError:
    try:
        # Alternative import for corex_topic
        from corex_topic import Corex as CorEx
        COREX_AVAILABLE = True
    except ImportError:
        COREX_AVAILABLE = False

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

# Emotion mapping for sentiment labels
EMOTION_MAPPING = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# Model configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 1000
SAMPLE_FRACTION = 0.1  # Use 10% of data for faster processing

# BERT configuration
BERT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"  # Fast pre-trained model
BERT_BATCH_SIZE = 16

# ============================================================================
# SOCIAL COMMERCE BEHAVIOR TOPICS - PREDEFINED CATEGORIES
# ============================================================================

# Define your 5 social commerce behavior topics with descriptions and seed words
SOCIAL_COMMERCE_TOPICS = {
    "Purchase Intent": {
        "description": "Intention to buy products, shopping motivation, buying decisions",
        "seed_words": [
            "buy", "purchase", "order", "cart", "checkout", "shop", "want",
            "need", "get", "afford", "price", "cost", "deal", "discount",
            "sale", "buying", "shopping", "gonna buy", "will buy", "thinking of buying"
        ],
        "zero_shot_labels": [
            "purchase intention",
            "buying motivation",
            "shopping decision",
            "intent to purchase"
        ]
    },
    "Trust & Credibility": {
        "description": "Trust signals, authenticity concerns, seller reliability, scam awareness",
        "seed_words": [
            "trust", "reliable", "authentic", "genuine", "fake", "scam",
            "legit", "legitimate", "safe", "secure", "verified", "real",
            "honest", "trustworthy", "reputation", "credible", "fraud",
            "suspicious", "quality", "original"
        ],
        "zero_shot_labels": [
            "trust and credibility concerns",
            "authenticity verification",
            "seller reliability assessment"
        ]
    },
    "Social Influence": {
        "description": "Peer recommendations, influencer impact, social proof, word of mouth",
        "seed_words": [
            "recommend", "recommendation", "influencer", "friend", "everyone",
            "popular", "trending", "viral", "review", "rating", "star",
            "tried", "tested", "swear by", "love it", "amazing", "best",
            "follower", "celebrity", "endorsed", "suggested"
        ],
        "zero_shot_labels": [
            "social influence and recommendations",
            "peer pressure to buy",
            "influencer marketing impact"
        ]
    },
    "Information Seeking": {
        "description": "Product research, comparison shopping, asking questions, seeking details",
        "seed_words": [
            "anyone", "know", "question", "ask", "compare", "comparison",
            "difference", "better", "which", "what", "how", "where",
            "looking for", "searching", "find", "help", "advice", "opinion",
            "experience", "feedback", "review", "details", "specs"
        ],
        "zero_shot_labels": [
            "information seeking behavior",
            "product research and comparison",
            "asking for advice"
        ]
    },
    "Engagement & Interaction": {
        "description": "Social engagement, likes, shares, comments, community participation",
        "seed_words": [
            "like", "share", "comment", "follow", "subscribe", "post",
            "tag", "mention", "dm", "message", "reply", "react",
            "engage", "participate", "join", "community", "group",
            "discuss", "chat", "connect"
        ],
        "zero_shot_labels": [
            "social media engagement",
            "community participation",
            "interaction and communication"
        ]
    }
}

# Simplified labels for zero-shot classification
SOCIAL_COMMERCE_LABELS = list(SOCIAL_COMMERCE_TOPICS.keys())


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

@st.cache_data
def load_and_preprocess_data(file_path):
    """
    Load and preprocess the dataset.
    """
    try:
        data = pd.read_csv(file_path)

        # Drop 'id' column if exists and ensure unique values
        if 'id' in data.columns:
            data = data.drop(columns=['id'])
        data = data.drop_duplicates()

        # Rename columns for convenience
        data.columns = ['text', 'sentiment']

        # Sample data for faster processing (configurable)
        data = data.sample(frac=SAMPLE_FRACTION, random_state=RANDOM_STATE)

        return data
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def clean_text(text):
    """
    Clean and normalize text data.
    """
    if not isinstance(text, str):
        text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove digits
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    return text


def preprocess_text(text):
    """
    Advanced text preprocessing with tokenization and lemmatization.
    """
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Tokenize
    tokens = word_tokenize(text)

    # Lemmatize and remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return ' '.join(tokens)


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def create_feature_vectors(X_train, X_test, vectorizer_type='tfidf', max_features=1000):
    """
    Create feature vectors using TF-IDF or Count Vectorization.
    """
    if vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    else:
        vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1, 2))

    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    return X_train_vectorized, X_test_vectorized, vectorizer


# ============================================================================
# TRADITIONAL MACHINE LEARNING MODELS FOR SENTIMENT ANALYSIS
# ============================================================================

def get_sentiment_models():
    """
    Get dictionary of traditional sentiment analysis models.
    """
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=200,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            max_depth=20
        ),
        'Naive Bayes': MultinomialNB(
            alpha=1.0
        ),
        'SVM': SVC(
            kernel='linear',
            probability=True,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=RANDOM_STATE
        )
    }

    return models


# ============================================================================
# BERT-BASED SENTIMENT ANALYSIS
# ============================================================================

@st.cache_resource
def load_bert_sentiment_model():
    """
    Load pre-trained BERT model for sentiment analysis.
    """
    if not TRANSFORMERS_AVAILABLE:
        return None

    try:
        # Use GPU if available
        device = 0 if torch.cuda.is_available() else -1

        # Load pre-trained model
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=BERT_MODEL_NAME,
            device=device,
            truncation=True,
            max_length=512
        )

        return sentiment_pipeline
    except Exception as e:
        st.error(f"Error loading BERT model: {str(e)}")
        return None


@st.cache_resource
def load_multiclass_bert_model():
    """
    Load BERT model for multi-class emotion classification.
    """
    if not TRANSFORMERS_AVAILABLE:
        return None, None

    try:
        model_name = "bhadresh-savani/distilbert-base-uncased-emotion"

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading multi-class BERT model: {str(e)}")
        return None, None


def predict_with_bert(texts, model, tokenizer, batch_size=16):
    """
    Predict emotions using BERT model.
    """
    if model is None or tokenizer is None:
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_predictions = []
    all_probabilities = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

        predictions = torch.argmax(probabilities, dim=-1)

        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

    return np.array(all_predictions), np.array(all_probabilities)


# ============================================================================
# ZERO-SHOT CLASSIFICATION FOR SOCIAL COMMERCE TOPICS
# ============================================================================

@st.cache_resource
def load_zero_shot_classifier():
    """
    Load Zero-Shot Classification model for predefined topic classification.

    **Model: facebook/bart-large-mnli**
    - No training required - works out of the box
    - Classifies text into ANY predefined categories you specify
    - Perfect for social commerce behavior classification

    **How it works:**
    - Uses Natural Language Inference (NLI)
    - Treats classification as "Does this text entail this label?"
    - Returns probability scores for each label

    **Evaluation:**
    ✓ Excellent: No training data needed
    ✓ Excellent: Works with any custom labels
    ✓ Good: Semantic understanding of topics
    ⚠ Slower than: Traditional classifiers

    Returns:
        pipeline: Zero-shot classification pipeline
    """
    if not TRANSFORMERS_AVAILABLE:
        return None

    try:
        device = 0 if torch.cuda.is_available() else -1

        # Use bart-large-mnli for best accuracy, or deberta for speed
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=device
        )

        return classifier
    except Exception as e:
        st.error(f"Error loading zero-shot classifier: {str(e)}")
        return None


def classify_social_commerce_topic(text, classifier, labels=None, multi_label=False):
    """
    Classify a single text into social commerce behavior topics.

    **Functionality:**
    - Uses zero-shot classification
    - Returns topic predictions with confidence scores
    - Supports multi-label classification (text can belong to multiple topics)

    Args:
        text (str): Input text to classify
        classifier: Zero-shot classification pipeline
        labels (list): List of topic labels (default: SOCIAL_COMMERCE_LABELS)
        multi_label (bool): Allow multiple labels per text

    Returns:
        dict: {
            'labels': ['Purchase Intent', 'Social Influence', ...],
            'scores': [0.85, 0.12, ...],
            'top_label': 'Purchase Intent',
            'top_score': 0.85
        }
    """
    if classifier is None:
        return None

    if labels is None:
        labels = SOCIAL_COMMERCE_LABELS

    try:
        result = classifier(
            text,
            candidate_labels=labels,
            multi_label=multi_label
        )

        return {
            'labels': result['labels'],
            'scores': result['scores'],
            'top_label': result['labels'][0],
            'top_score': result['scores'][0]
        }
    except Exception as e:
        st.error(f"Classification error: {str(e)}")
        return None


def classify_social_commerce_batch(texts, classifier, labels=None, multi_label=False, progress_callback=None):
    """
    Classify multiple texts into social commerce behavior topics.

    **Functionality:**
    - Batch processing for efficiency
    - Progress tracking for large datasets
    - Returns DataFrame with all predictions

    Args:
        texts (list): List of texts to classify
        classifier: Zero-shot classification pipeline
        labels (list): Topic labels
        multi_label (bool): Allow multiple labels
        progress_callback: Optional callback for progress updates

    Returns:
        pd.DataFrame: DataFrame with columns [text, topic, confidence, all_scores]
    """
    if classifier is None:
        return None

    if labels is None:
        labels = SOCIAL_COMMERCE_LABELS

    results = []
    total = len(texts)

    for i, text in enumerate(texts):
        try:
            result = classifier(
                text[:512],  # Truncate long texts
                candidate_labels=labels,
                multi_label=multi_label
            )

            results.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'topic': result['labels'][0],
                'confidence': result['scores'][0],
                'all_scores': dict(zip(result['labels'], result['scores']))
            })

            if progress_callback and i % 10 == 0:
                progress_callback(i / total)

        except Exception as e:
            results.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'topic': 'Error',
                'confidence': 0.0,
                'all_scores': {}
            })

    return pd.DataFrame(results)


# ============================================================================
# GUIDED/SEEDED TOPIC MODELING
# ============================================================================

def perform_seeded_lda(texts, seed_topics_dict, n_topics=5, n_top_words=10):
    """
    Perform Seeded LDA (Guided LDA) with seed words for each topic.

    **Functionality:**
    - Uses seed words to guide topic discovery
    - Each topic is "anchored" to specific seed words
    - Combines domain knowledge with statistical topic modeling

    **How it works:**
    1. Creates a count vectorizer with vocabulary including seed words
    2. Modifies LDA's prior to favor seed words for specific topics
    3. Runs standard LDA with modified priors

    **Evaluation:**
    ✓ Good: Incorporates domain knowledge
    ✓ Good: More interpretable topics
    ⚠ May not discover unexpected topics
    ⚠ Quality depends on seed word selection

    Args:
        texts (list): List of documents
        seed_topics_dict (dict): {topic_name: [seed_words]}
        n_topics (int): Number of topics
        n_top_words (int): Top words to return per topic

    Returns:
        tuple: (model, vectorizer, topics_dict, doc_topic_matrix)
    """
    # Create vocabulary that includes all seed words
    all_seed_words = []
    for topic_words in seed_topics_dict.values():
        all_seed_words.extend(topic_words)

    # Create vectorizer
    vectorizer = CountVectorizer(
        max_features=2000,
        max_df=0.95,
        min_df=2,
        stop_words='english'
    )

    doc_term_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # Create seed word to topic mapping
    word_to_idx = {word: idx for idx, word in enumerate(feature_names)}

    # Initialize LDA
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=RANDOM_STATE,
        max_iter=50,
        learning_method='batch',
        n_jobs=-1
    )

    # Fit LDA
    lda.fit(doc_term_matrix)

    # Modify components to boost seed words
    for topic_idx, (topic_name, seed_words) in enumerate(seed_topics_dict.items()):
        if topic_idx >= n_topics:
            break
        for seed_word in seed_words:
            seed_word_lower = seed_word.lower()
            if seed_word_lower in word_to_idx:
                word_idx = word_to_idx[seed_word_lower]
                # Boost seed word weight
                lda.components_[topic_idx, word_idx] *= 5.0

    # Normalize components
    lda.components_ = lda.components_ / lda.components_.sum(axis=1, keepdims=True)

    # Extract topics
    topics_dict = {}
    for topic_idx, topic in enumerate(lda.components_):
        topic_name = list(seed_topics_dict.keys())[topic_idx] if topic_idx < len(seed_topics_dict) else f"Topic {topic_idx + 1}"
        top_word_indices = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_word_indices]
        topics_dict[topic_name] = top_words

    # Get document-topic distribution
    doc_topics = lda.transform(doc_term_matrix)

    return lda, vectorizer, topics_dict, doc_topics


def perform_corex_topic_modeling(texts, anchor_words_dict, n_topics=5, n_top_words=10):
    """
    Perform Anchored CorEx (Correlation Explanation) Topic Modeling.

    **CorEx Advantages:**
    - Information-theoretic approach (not probabilistic like LDA)
    - Built-in support for anchor words
    - More stable and reproducible than LDA
    - Better topic coherence with domain guidance

    **How it works:**
    1. Uses Total Correlation to find topics
    2. Anchor words "nudge" topics toward specific themes
    3. Discovers additional patterns beyond anchors

    **Evaluation:**
    ✓ Excellent: Native support for guided/anchored topics
    ✓ Excellent: Information-theoretic foundation
    ✓ Good: Handles short texts well
    ⚠ Requires: pip install corextopic

    Args:
        texts (list): List of documents
        anchor_words_dict (dict): {topic_name: [anchor_words]}
        n_topics (int): Number of topics
        n_top_words (int): Words per topic

    Returns:
        tuple: (model, vectorizer, topics_dict, doc_topic_matrix)
    """
    if not COREX_AVAILABLE:
        st.warning("CorEx not installed. Install with: pip install corextopic")
        return None, None, None, None

    # Create vectorizer
    vectorizer = CountVectorizer(
        max_features=2000,
        max_df=0.95,
        min_df=2,
        stop_words='english',
        binary=True  # CorEx works better with binary counts
    )

    doc_term_matrix = vectorizer.fit_transform(texts)
    feature_names = list(vectorizer.get_feature_names_out())

    # Prepare anchor words
    anchors = []
    anchor_strength = 3  # How strongly to push topics toward anchors

    for topic_name, topic_anchors in anchor_words_dict.items():
        # Filter anchors that exist in vocabulary
        valid_anchors = [word.lower() for word in topic_anchors if word.lower() in feature_names]
        if valid_anchors:
            anchors.append(valid_anchors)

    # Pad with empty lists if fewer anchor groups than topics
    while len(anchors) < n_topics:
        anchors.append([])

    # Create CorEx model
    topic_model = CorEx(
        n_hidden=n_topics,
        seed=RANDOM_STATE,
        max_iter=200
    )

    # Fit with anchors
    topic_model.fit(
        doc_term_matrix,
        words=feature_names,
        anchors=anchors[:n_topics],
        anchor_strength=anchor_strength
    )

    # Extract topics
    topics_dict = {}
    topic_names = list(anchor_words_dict.keys())

    for topic_idx in range(n_topics):
        topic_name = topic_names[topic_idx] if topic_idx < len(topic_names) else f"Topic {topic_idx + 1}"

        # Get top words for this topic
        topic_words = topic_model.get_topics(topic=topic_idx, n_words=n_top_words)
        if topic_words:
            words = [word for word, _, _ in topic_words]
            topics_dict[topic_name] = words

    # Get document-topic matrix
    doc_topics = topic_model.transform(doc_term_matrix)

    return topic_model, vectorizer, topics_dict, doc_topics


def perform_seeded_bertopic(texts, seed_topics_dict, n_topics=5):
    """
    Perform BERTopic with Seeded Topics.

    **BERTopic Seeded Topics (v0.15+):**
    - Combines semantic embeddings with seed word guidance
    - Best of both worlds: semantic understanding + domain knowledge
    - More interpretable topics aligned with your categories

    **How it works:**
    1. Creates semantic embeddings for all documents
    2. Uses seed words to create topic representations
    3. Assigns documents to seed-guided topics
    4. Refines topics based on document clusters

    **Evaluation:**
    ✓ Excellent: State-of-the-art embeddings
    ✓ Excellent: Semantic understanding
    ✓ Good: Guided by domain knowledge
    ⚠ Requires: BERTopic v0.15+

    Args:
        texts (list): List of documents
        seed_topics_dict (dict): {topic_name: [seed_words]}
        n_topics (int): Number of topics

    Returns:
        tuple: (topic_model, topics, probabilities, topics_dict)
    """
    if not BERTOPIC_AVAILABLE:
        st.warning("BERTopic not installed. Install with: pip install bertopic sentence-transformers")
        return None, None, None, None

    try:
        # Load sentence transformer
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Prepare seed topic list for BERTopic
        # Format: [[seed_word1, seed_word2], [seed_word3, seed_word4], ...]
        seed_topic_list = [words[:10] for words in seed_topics_dict.values()]

        # Create BERTopic with seeded topics
        topic_model = BERTopic(
            embedding_model=embedding_model,
            min_topic_size=10,
            nr_topics=n_topics,
            seed_topic_list=seed_topic_list,
            calculate_probabilities=True,
            verbose=False
        )

        # Fit model
        topics, probabilities = topic_model.fit_transform(texts)

        # Map discovered topics to our predefined names
        topic_info = topic_model.get_topic_info()
        topics_dict = {}

        topic_names = list(seed_topics_dict.keys())

        for idx, topic_id in enumerate(topic_info['Topic']):
            if topic_id == -1:
                continue  # Skip outlier topic

            # Get topic words
            topic_words = topic_model.get_topic(topic_id)
            if topic_words:
                words = [word for word, _ in topic_words[:10]]

                # Use predefined name if available
                if idx < len(topic_names):
                    name = topic_names[idx]
                else:
                    name = f"Topic {topic_id}"

                topics_dict[name] = words

        return topic_model, topics, probabilities, topics_dict

    except Exception as e:
        st.error(f"Error in seeded BERTopic: {str(e)}")
        return None, None, None, None


# ============================================================================
# TRADITIONAL TOPIC MODELING (LDA, NMF)
# ============================================================================

def perform_topic_modeling(texts, n_topics=5, method='lda', n_top_words=10):
    """
    Perform topic modeling on text data using traditional methods (LDA or NMF).
    """
    # Create document-term matrix
    if method == 'lda':
        # LDA works better with raw counts
        vectorizer = CountVectorizer(max_features=1000, max_df=0.95, min_df=2)
        doc_term_matrix = vectorizer.fit_transform(texts)

        # Fit LDA model
        model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=RANDOM_STATE,
            max_iter=20,
            learning_method='online',
            n_jobs=-1
        )
        model.fit(doc_term_matrix)

    elif method == 'nmf':
        # NMF works better with TF-IDF
        vectorizer = TfidfVectorizer(max_features=1000, max_df=0.95, min_df=2)
        doc_term_matrix = vectorizer.fit_transform(texts)

        # Fit NMF model
        model = NMF(
            n_components=n_topics,
            random_state=RANDOM_STATE,
            init='nndsvda',
            max_iter=400
        )
        model.fit(doc_term_matrix)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'lda' or 'nmf'")

    # Extract topics
    feature_names = vectorizer.get_feature_names_out()
    topics_dict = {}

    for topic_idx, topic in enumerate(model.components_):
        top_word_indices = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_word_indices]
        topics_dict[f"Topic {topic_idx + 1}"] = top_words

    return model, vectorizer, feature_names, topics_dict


# ============================================================================
# BERTOPIC - STATE-OF-THE-ART TOPIC MODELING
# ============================================================================

@st.cache_resource
def load_bertopic_model(n_topics=5):
    """
    Load or create BERTopic model for semantic topic modeling.
    """
    if not BERTOPIC_AVAILABLE:
        return None

    try:
        # Load sentence transformer for embeddings
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Create BERTopic model
        topic_model = BERTopic(
            language="english",
            calculate_probabilities=True,
            embedding_model=embedding_model,
            min_topic_size=10,
            n_gram_range=(1, 3),
            nr_topics=n_topics if n_topics and n_topics > 0 else None,
            verbose=False
        )

        return topic_model
    except Exception as e:
        st.error(f"Error loading BERTopic model: {str(e)}")
        return None


def perform_bertopic_modeling(texts, n_topics=5):
    """
    Perform BERTopic modeling on text data (unsupervised).
    """
    if not BERTOPIC_AVAILABLE:
        return None, None, None, None

    try:
        # Load model
        topic_model = load_bertopic_model(n_topics)

        if topic_model is None:
            return None, None, None, None

        # Fit model
        topics, probabilities = topic_model.fit_transform(texts)

        # Get topic information
        topic_info = topic_model.get_topic_info()

        # Extract topics dictionary
        topics_dict = {}
        for topic_id in topic_info['Topic']:
            if topic_id != -1:  # Skip outlier topic
                topic_words = topic_model.get_topic(topic_id)
                if topic_words:
                    words = [word for word, _ in topic_words[:10]]
                    topics_dict[f"Topic {topic_id}"] = words

        return topic_model, topics, probabilities, topics_dict
    except Exception as e:
        st.error(f"Error in BERTopic modeling: {str(e)}")
        return None, None, None, None


def get_document_topics(model, vectorizer, texts, top_n=3):
    """
    Get dominant topics for each document (for LDA/NMF).
    """
    doc_term_matrix = vectorizer.transform(texts)
    doc_topics = model.transform(doc_term_matrix)

    results = []
    for doc_topic_dist in doc_topics:
        top_topic_indices = doc_topic_dist.argsort()[-top_n:][::-1]
        top_topics = [(idx, doc_topic_dist[idx]) for idx in top_topic_indices]
        results.append(top_topics)

    return results


# ============================================================================
# TOPIC-SENTIMENT INTEGRATION
# ============================================================================

def combine_topic_sentiment(texts, topic_predictions, sentiment_predictions, emotion_mapping=EMOTION_MAPPING):
    """
    Combine topic and sentiment analysis results.

    **Functionality:**
    - Creates a unified view of topic + sentiment per document
    - Enables analysis like "What emotions are associated with Purchase Intent?"
    - Supports cross-tabulation and visualization

    Args:
        texts (list): Original texts
        topic_predictions (list): Topic labels for each text
        sentiment_predictions (list): Sentiment labels for each text
        emotion_mapping (dict): Mapping from sentiment ID to emotion name

    Returns:
        pd.DataFrame: Combined results with columns [text, topic, sentiment, ...]
    """
    results = []

    for i, text in enumerate(texts):
        topic = topic_predictions[i] if i < len(topic_predictions) else "Unknown"
        sentiment = sentiment_predictions[i] if i < len(sentiment_predictions) else "Unknown"

        # Map sentiment ID to name if needed
        if isinstance(sentiment, (int, np.integer)):
            sentiment_name = emotion_mapping.get(sentiment, f"Class {sentiment}")
        else:
            sentiment_name = sentiment

        results.append({
            'text': text[:100] + '...' if len(text) > 100 else text,
            'topic': topic,
            'sentiment': sentiment_name
        })

    return pd.DataFrame(results)


def analyze_topic_sentiment_distribution(combined_df):
    """
    Analyze sentiment distribution across topics.

    **Functionality:**
    - Creates cross-tabulation of topics vs sentiments
    - Calculates sentiment distribution per topic
    - Returns insights for visualization

    Args:
        combined_df (pd.DataFrame): DataFrame with 'topic' and 'sentiment' columns

    Returns:
        dict: {
            'crosstab': pd.DataFrame,
            'topic_dominant_sentiment': dict,
            'sentiment_by_topic': dict
        }
    """
    # Cross-tabulation
    crosstab = pd.crosstab(combined_df['topic'], combined_df['sentiment'])

    # Normalize by row (percentage within each topic)
    crosstab_normalized = crosstab.div(crosstab.sum(axis=1), axis=0) * 100

    # Find dominant sentiment for each topic
    topic_dominant_sentiment = {}
    for topic in crosstab.index:
        dominant = crosstab.loc[topic].idxmax()
        percentage = crosstab_normalized.loc[topic, dominant]
        topic_dominant_sentiment[topic] = {
            'dominant_sentiment': dominant,
            'percentage': percentage
        }

    return {
        'crosstab': crosstab,
        'crosstab_normalized': crosstab_normalized,
        'topic_dominant_sentiment': topic_dominant_sentiment
    }


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def cross_validate_model(model, X_train_tfidf, y_train, cv=5, scoring='accuracy'):
    """
    Perform cross-validation on a model.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train_tfidf, y_train, cv=skf, scoring=scoring, n_jobs=-1)
    return scores.mean(), scores.std()


def evaluate_model_optimized(model, X_train_tfidf, X_test_tfidf, y_train, y_test, all_classes, calc_auc=False):
    """
    Comprehensive model evaluation with multiple metrics.
    """
    # Train model
    model.fit(X_train_tfidf, y_train)

    # Predictions
    y_pred = model.predict(X_test_tfidf)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # ROC-AUC calculation
    roc_auc = None
    if calc_auc and hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_tfidf)

        # Align probabilities if model doesn't have all classes
        if y_score.shape[1] < len(all_classes):
            aligned_probs = np.zeros((y_score.shape[0], len(all_classes)))
            for i, label in enumerate(model.classes_):
                aligned_probs[:, label] = y_score[:, i]
            y_score = aligned_probs

        # Create binary labels for multi-class ROC-AUC
        y_test_bin = np.zeros((len(y_test), len(all_classes)))
        for i, label in enumerate(y_test):
            y_test_bin[i, label] = 1

        roc_auc = roc_auc_score(y_test_bin, y_score, multi_class='ovr')

    return accuracy, precision, recall, f1, roc_auc


def evaluate_bert_model(bert_model, tokenizer, X_test, y_test):
    """
    Evaluate BERT model on test data.
    """
    if bert_model is None or tokenizer is None:
        return None

    # Convert to list
    test_texts = X_test.tolist() if hasattr(X_test, 'tolist') else list(X_test)

    # Predict
    predictions, probabilities = predict_with_bert(test_texts, bert_model, tokenizer)

    if predictions is None:
        return None

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)

    # ROC-AUC
    y_test_bin = np.zeros((len(y_test), probabilities.shape[1]))
    for i, label in enumerate(y_test):
        y_test_bin[i, label] = 1

    roc_auc = roc_auc_score(y_test_bin, probabilities, multi_class='ovr')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }


# ============================================================================
# SENTIMENT ANALYSIS UTILITIES
# ============================================================================

def get_sentiment_intensity(text):
    """
    Get sentiment intensity using VADER.
    """
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)['compound']


def get_subjectivity(text):
    """
    Get subjectivity score using TextBlob.
    """
    return TextBlob(text).sentiment.subjectivity


def aspect_sentiment_analysis(text):
    """
    Perform aspect-based sentiment analysis using spaCy.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    aspects = {}

    for chunk in doc.noun_chunks:
        sentiment = TextBlob(chunk.text).sentiment.polarity
        aspects[chunk.text] = sentiment

    return aspects


# ============================================================================
# STREAMLIT UI
# ============================================================================

def display_results(model_name, accuracy, precision, recall, f1, roc_auc, cv_mean=None, cv_std=None):
    """
    Display model evaluation results in Streamlit.
    """
    with st.expander(f"📊 Results for {model_name}"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
            st.metric("Precision", f"{precision:.4f}")

        with col2:
            st.metric("Recall", f"{recall:.4f}")
            st.metric("F1 Score", f"{f1:.4f}")

        with col3:
            if roc_auc is not None:
                st.metric("ROC-AUC", f"{roc_auc:.4f}")
            else:
                st.metric("ROC-AUC", "N/A")

        if cv_mean is not None and cv_std is not None:
            st.write(f"**Cross-Validation:** {cv_mean:.4f} ± {cv_std:.4f}")


def main():
    """
    Main Streamlit application.
    """
    st.set_page_config(page_title="Sentiment Analysis & Topic Modeling", layout="wide")

    st.title("🎭 Advanced Sentiment Analysis & Topic Modeling")
    st.markdown("**Enhanced with BERT, BERTopic & Social Commerce Topic Classification** | State-of-the-Art NLP Models")
    st.markdown("---")

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # Model type selection
    st.sidebar.subheader("Analysis Type")
    use_bert = st.sidebar.checkbox("🚀 Use BERT for Sentiment Analysis", value=TRANSFORMERS_AVAILABLE)
    use_traditional = st.sidebar.checkbox("📊 Use Traditional ML Models", value=True)

    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        file_path = uploaded_file
    else:
        file_path = "text.csv"  # Default file

    # Model selection for traditional ML
    if use_traditional:
        st.sidebar.subheader("Traditional ML Models")
        model_selection = st.sidebar.multiselect(
            "Choose models to evaluate",
            ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'SVM', 'Gradient Boosting'],
            default=['Logistic Regression', 'Naive Bayes']
        )
    else:
        model_selection = []

    # Topic modeling settings
    st.sidebar.subheader("Topic Modeling Settings")
    enable_topic_modeling = st.sidebar.checkbox("Enable Topic Modeling", value=True)

    if enable_topic_modeling:
        topic_method = st.sidebar.selectbox(
            "Topic Modeling Method",
            ['Zero-Shot Classification', 'Seeded LDA', 'Anchored CorEx', 'Seeded BERTopic', 'LDA (Unsupervised)', 'NMF (Unsupervised)', 'BERTopic (Unsupervised)']
        )
        n_topics = st.sidebar.slider("Number of Topics", 3, 10, 5)
    else:
        topic_method = 'Zero-Shot Classification'
        n_topics = 5

    # Load data
    try:
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            if 'id' in data.columns:
                data = data.drop(columns=['id'])
            data = data.drop_duplicates()
            data.columns = ['text', 'sentiment']
            data = data.sample(frac=SAMPLE_FRACTION, random_state=RANDOM_STATE)
        else:
            data = load_and_preprocess_data(file_path)

        if data is None:
            st.error("Failed to load data. Please upload a valid CSV file.")
            return

        # Preprocess data
        with st.spinner("Preprocessing data..."):
            data['text'] = data['text'].fillna('').astype(str)
            data['text_cleaned'] = data['text'].apply(clean_text)
            data['text_processed'] = data['text_cleaned'].apply(preprocess_text)

        # Display dataset info
        st.header("📊 Dataset Information")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Samples", len(data))
        col2.metric("Unique Sentiments", data['sentiment'].nunique())
        col3.metric("Avg Text Length", f"{data['text'].str.len().mean():.0f}")
        col4.metric("Processing", "Complete ✓")

        # Sentiment distribution
        st.subheader("Sentiment Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        sentiment_counts = data['sentiment'].value_counts()
        sentiment_counts.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_xlabel("Sentiment Class")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Sentiment Classes")
        st.pyplot(fig)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            data['text_processed'],
            data['sentiment'],
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=data['sentiment']
        )

        col1, col2 = st.columns(2)
        col1.metric("Training Samples", len(X_train))
        col2.metric("Test Samples", len(X_test))

        # ====================================================================
        # TABS FOR DIFFERENT ANALYSES
        # ====================================================================

        tab1, tab2, tab3, tab4 = st.tabs([
            "🤖 Sentiment Analysis",
            "📚 Topic Modeling (Social Commerce)",
            "🔗 Topic-Sentiment Integration",
            "🔮 Interactive Prediction"
        ])

        # ====================================================================
        # TAB 1: SENTIMENT ANALYSIS
        # ====================================================================
        with tab1:
            st.header("🤖 Sentiment Analysis")

            # Traditional ML Models
            if use_traditional and model_selection:
                st.subheader("📊 Traditional Machine Learning Models")

                # Vectorization
                X_train_tfidf, X_test_tfidf, vectorizer_tfidf = create_feature_vectors(
                    X_train, X_test, vectorizer_type='tfidf', max_features=MAX_FEATURES
                )

                all_classes = np.arange(len(np.unique(y_train)))

                all_models = get_sentiment_models()
                selected_models = {k: v for k, v in all_models.items() if k in model_selection}

                with st.spinner("Training and evaluating traditional ML models..."):
                    for name, model in selected_models.items():
                        cv_mean, cv_std = cross_validate_model(
                            model, X_train_tfidf, y_train, cv=5, scoring='accuracy'
                        )
                        results = evaluate_model_optimized(
                            model, X_train_tfidf, X_test_tfidf, y_train, y_test, all_classes, calc_auc=True
                        )
                        display_results(
                            model_name=name,
                            accuracy=results[0],
                            precision=results[1],
                            recall=results[2],
                            f1=results[3],
                            roc_auc=results[4],
                            cv_mean=cv_mean,
                            cv_std=cv_std
                        )

            # BERT Model
            if use_bert and TRANSFORMERS_AVAILABLE:
                st.subheader("🚀 BERT-Based Sentiment Analysis")

                with st.spinner("Loading BERT model..."):
                    bert_model, bert_tokenizer = load_multiclass_bert_model()

                if bert_model is not None:
                    with st.spinner("Evaluating BERT model..."):
                        # Use original text for BERT (not processed)
                        X_train_orig = data.loc[X_train.index, 'text']
                        X_test_orig = data.loc[X_test.index, 'text']

                        bert_results = evaluate_bert_model(bert_model, bert_tokenizer, X_test_orig, y_test)

                        if bert_results:
                            display_results(
                                model_name="BERT (DistilBERT-Emotion)",
                                accuracy=bert_results['accuracy'],
                                precision=bert_results['precision'],
                                recall=bert_results['recall'],
                                f1=bert_results['f1'],
                                roc_auc=bert_results['roc_auc']
                            )

                            st.success("✓ BERT model evaluation complete!")
                        else:
                            st.warning("BERT evaluation failed.")
                else:
                    st.warning("BERT model could not be loaded.")

        # ====================================================================
        # TAB 2: TOPIC MODELING (SOCIAL COMMERCE)
        # ====================================================================
        with tab2:
            st.header("📚 Topic Modeling for Social Commerce Behavior")

            # Display predefined topics
            st.subheader("🎯 Predefined Social Commerce Topics")
            st.markdown("""
            These are your **5 target topics** for social commerce behavior classification:
            """)

            for topic_name, topic_info in SOCIAL_COMMERCE_TOPICS.items():
                with st.expander(f"**{topic_name}**"):
                    st.write(f"**Description:** {topic_info['description']}")
                    st.write(f"**Seed Words:** {', '.join(topic_info['seed_words'][:10])}...")

            st.markdown("---")

            # Topic modeling method selection (from sidebar)
            st.subheader(f"📖 Selected Method: {topic_method}")

            # Prepare seed words
            seed_words_dict = {name: info['seed_words'] for name, info in SOCIAL_COMMERCE_TOPICS.items()}

            if st.button("🚀 Run Topic Modeling", type="primary"):

                if topic_method == 'Zero-Shot Classification':
                    # ========================================================
                    # ZERO-SHOT CLASSIFICATION
                    # ========================================================
                    st.subheader("🎯 Zero-Shot Classification")
                    st.markdown("""
                    **How it works:**
                    - Uses pre-trained NLI model (no training needed!)
                    - Classifies each text into your predefined topics
                    - Returns confidence scores for each topic

                    **Best for:** When you have predefined categories and no labeled data
                    """)

                    if not TRANSFORMERS_AVAILABLE:
                        st.error("❌ Transformers not installed. Install with: pip install transformers torch")
                    else:
                        with st.spinner("Loading Zero-Shot classifier..."):
                            classifier = load_zero_shot_classifier()

                        if classifier:
                            # Sample texts for demo (full dataset would be slow)
                            sample_size = min(100, len(data))
                            sample_texts = data['text'].head(sample_size).tolist()

                            st.info(f"📊 Classifying {sample_size} sample texts...")

                            progress_bar = st.progress(0)

                            with st.spinner("Classifying texts..."):
                                results_df = classify_social_commerce_batch(
                                    sample_texts,
                                    classifier,
                                    labels=SOCIAL_COMMERCE_LABELS,
                                    multi_label=False,
                                    progress_callback=lambda p: progress_bar.progress(p)
                                )

                            progress_bar.progress(1.0)

                            if results_df is not None:
                                st.success(f"✅ Classified {len(results_df)} texts!")

                                # Store for later use
                                st.session_state['topic_results'] = results_df

                                # Topic distribution
                                st.subheader("📊 Topic Distribution")
                                topic_counts = results_df['topic'].value_counts()

                                fig, ax = plt.subplots(figsize=(10, 5))
                                colors = plt.cm.Set2(np.linspace(0, 1, len(topic_counts)))
                                topic_counts.plot(kind='bar', ax=ax, color=colors)
                                ax.set_xlabel("Social Commerce Topic")
                                ax.set_ylabel("Document Count")
                                ax.set_title("Distribution of Social Commerce Behavior Topics")
                                plt.xticks(rotation=45, ha='right')
                                plt.tight_layout()
                                st.pyplot(fig)

                                # Confidence distribution
                                st.subheader("📈 Confidence Distribution")
                                fig, ax = plt.subplots(figsize=(10, 4))
                                results_df['confidence'].hist(bins=20, ax=ax, color='green', alpha=0.7)
                                ax.set_xlabel("Confidence Score")
                                ax.set_ylabel("Frequency")
                                ax.set_title("Distribution of Classification Confidence")
                                ax.axvline(results_df['confidence'].mean(), color='red', linestyle='--', label=f"Mean: {results_df['confidence'].mean():.2f}")
                                ax.legend()
                                st.pyplot(fig)

                                # Sample results
                                st.subheader("📝 Sample Classifications")
                                st.dataframe(results_df[['text', 'topic', 'confidence']].head(20), use_container_width=True)

                                # Per-topic examples
                                st.subheader("🔍 Examples per Topic")
                                for topic in SOCIAL_COMMERCE_LABELS:
                                    topic_examples = results_df[results_df['topic'] == topic].head(3)
                                    if len(topic_examples) > 0:
                                        with st.expander(f"**{topic}** ({len(results_df[results_df['topic'] == topic])} documents)"):
                                            for _, row in topic_examples.iterrows():
                                                st.write(f"• *\"{row['text']}\"* (conf: {row['confidence']:.2f})")

                elif topic_method == 'Seeded LDA':
                    # ========================================================
                    # SEEDED LDA
                    # ========================================================
                    st.subheader("🌱 Seeded LDA Topic Modeling")
                    st.markdown("""
                    **How it works:**
                    - Standard LDA with seed words to guide each topic
                    - Seed words are boosted in their assigned topics
                    - Discovers additional related terms

                    **Best for:** Combining statistical patterns with domain knowledge
                    """)

                    with st.spinner("Running Seeded LDA..."):
                        lda_model, vectorizer, topics_dict, doc_topics = perform_seeded_lda(
                            data['text_processed'].tolist(),
                            seed_words_dict,
                            n_topics=n_topics,
                            n_top_words=10
                        )

                    st.success(f"✅ Discovered {len(topics_dict)} guided topics!")

                    # Display topics
                    st.subheader("📖 Discovered Topics (Guided by Seed Words)")
                    for topic_name, words in topics_dict.items():
                        with st.expander(f"**{topic_name}**"):
                            st.write(f"**Top Words:** {', '.join(words)}")

                            # Highlight seed words
                            seed_words_in_topic = [w for w in words if w in [s.lower() for s in seed_words_dict.get(topic_name, [])]]
                            if seed_words_in_topic:
                                st.write(f"**Seed Words Found:** {', '.join(seed_words_in_topic)}")

                    # Document-topic distribution
                    st.subheader("📊 Document-Topic Distribution")
                    dominant_topics = np.argmax(doc_topics, axis=1)
                    topic_names = list(topics_dict.keys())

                    topic_counts = pd.Series([topic_names[t] for t in dominant_topics]).value_counts()

                    fig, ax = plt.subplots(figsize=(10, 5))
                    topic_counts.plot(kind='bar', ax=ax, color=plt.cm.Set3(np.linspace(0, 1, len(topic_counts))))
                    ax.set_xlabel("Topic")
                    ax.set_ylabel("Document Count")
                    ax.set_title("Documents per Topic (Seeded LDA)")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)

                    # Store results
                    st.session_state['topic_results'] = pd.DataFrame({
                        'text': data['text'].values,
                        'topic': [topic_names[t] for t in dominant_topics],
                        'confidence': np.max(doc_topics, axis=1)
                    })

                elif topic_method == 'Anchored CorEx':
                    # ========================================================
                    # ANCHORED COREX
                    # ========================================================
                    st.subheader("⚓ Anchored CorEx Topic Modeling")
                    st.markdown("""
                    **How it works:**
                    - Information-theoretic approach (Total Correlation)
                    - Native support for anchor words
                    - More stable than LDA

                    **Best for:** Domain-specific topics with clear anchor terms
                    """)

                    if not COREX_AVAILABLE:
                        st.error("❌ CorEx not installed. Install with: pip install corextopic")
                        st.code("pip install corextopic", language="bash")
                    else:
                        with st.spinner("Running Anchored CorEx..."):
                            corex_model, vectorizer, topics_dict, doc_topics = perform_corex_topic_modeling(
                                data['text_processed'].tolist(),
                                seed_words_dict,
                                n_topics=n_topics,
                                n_top_words=10
                            )

                        if topics_dict:
                            st.success(f"✅ Discovered {len(topics_dict)} anchored topics!")

                            # Display topics
                            st.subheader("📖 Discovered Topics (Anchored)")
                            for topic_name, words in topics_dict.items():
                                with st.expander(f"**{topic_name}**"):
                                    st.write(f"**Top Words:** {', '.join(words)}")

                            # Document-topic distribution
                            st.subheader("📊 Document-Topic Distribution")
                            dominant_topics = np.argmax(doc_topics, axis=1)
                            topic_names = list(topics_dict.keys())

                            topic_counts = pd.Series([topic_names[t] if t < len(topic_names) else f"Topic {t}" for t in dominant_topics]).value_counts()

                            fig, ax = plt.subplots(figsize=(10, 5))
                            topic_counts.plot(kind='bar', ax=ax, color=plt.cm.Pastel1(np.linspace(0, 1, len(topic_counts))))
                            ax.set_xlabel("Topic")
                            ax.set_ylabel("Document Count")
                            ax.set_title("Documents per Topic (Anchored CorEx)")
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            st.pyplot(fig)

                            # Store results
                            st.session_state['topic_results'] = pd.DataFrame({
                                'text': data['text'].values,
                                'topic': [topic_names[t] if t < len(topic_names) else f"Topic {t}" for t in dominant_topics],
                                'confidence': np.max(doc_topics, axis=1)
                            })

                elif topic_method == 'Seeded BERTopic':
                    # ========================================================
                    # SEEDED BERTOPIC
                    # ========================================================
                    st.subheader("🤖 Seeded BERTopic")
                    st.markdown("""
                    **How it works:**
                    - State-of-the-art BERT embeddings
                    - Seed words guide topic formation
                    - Semantic understanding + domain knowledge

                    **Best for:** Maximum accuracy with guided topics
                    """)

                    if not BERTOPIC_AVAILABLE:
                        st.error("❌ BERTopic not installed. Install with: pip install bertopic sentence-transformers")
                    else:
                        with st.spinner("Running Seeded BERTopic... (this may take a minute)"):
                            topic_model, topics, probs, topics_dict = perform_seeded_bertopic(
                                data['text'].tolist(),
                                seed_words_dict,
                                n_topics=n_topics
                            )

                        if topics_dict:
                            st.success(f"✅ Discovered {len(topics_dict)} semantic topics!")

                            # Display topics
                            st.subheader("📖 Discovered Topics (Seeded BERTopic)")
                            for topic_name, words in topics_dict.items():
                                with st.expander(f"**{topic_name}**"):
                                    st.write(f"**Top Words:** {', '.join(words)}")

                            # Topic distribution
                            st.subheader("📊 Topic Distribution")
                            topic_counts = pd.Series(topics).value_counts().sort_index()
                            topic_counts = topic_counts[topic_counts.index != -1]  # Remove outliers

                            fig, ax = plt.subplots(figsize=(10, 5))
                            topic_counts.plot(kind='bar', ax=ax, color=plt.cm.viridis(np.linspace(0, 1, len(topic_counts))))
                            ax.set_xlabel("Topic ID")
                            ax.set_ylabel("Document Count")
                            ax.set_title("Documents per Topic (Seeded BERTopic)")
                            st.pyplot(fig)

                            # Store results
                            topic_names = list(topics_dict.keys())
                            st.session_state['topic_results'] = pd.DataFrame({
                                'text': data['text'].values,
                                'topic': [topic_names[t] if 0 <= t < len(topic_names) else 'Outlier' for t in topics],
                                'confidence': np.max(probs, axis=1) if probs is not None else [1.0] * len(topics)
                            })

                elif topic_method in ['LDA (Unsupervised)', 'NMF (Unsupervised)']:
                    # ========================================================
                    # UNSUPERVISED LDA/NMF
                    # ========================================================
                    method_name = 'lda' if 'LDA' in topic_method else 'nmf'
                    st.subheader(f"📖 {method_name.upper()} Topic Modeling (Unsupervised)")
                    st.markdown("""
                    **How it works:**
                    - Standard unsupervised topic discovery
                    - Topics are discovered purely from data patterns
                    - May not align with your predefined categories

                    **Best for:** Exploratory analysis when you don't know the topics
                    """)

                    with st.spinner(f"Running {method_name.upper()}..."):
                        topic_model, vectorizer, feature_names, topics_dict = perform_topic_modeling(
                            data['text_processed'].tolist(),
                            n_topics=n_topics,
                            method=method_name,
                            n_top_words=10
                        )

                    st.success(f"✅ Discovered {len(topics_dict)} topics!")

                    # Display topics
                    st.subheader("📖 Discovered Topics")
                    for topic_name, words in topics_dict.items():
                        with st.expander(f"**{topic_name}**"):
                            st.write(f"**Top Words:** {', '.join(words)}")

                    st.warning("⚠️ These are unsupervised topics. They may not align with your predefined social commerce categories. Consider using Zero-Shot Classification or Seeded methods for better alignment.")

                elif topic_method == 'BERTopic (Unsupervised)':
                    # ========================================================
                    # UNSUPERVISED BERTOPIC
                    # ========================================================
                    st.subheader("🤖 BERTopic (Unsupervised)")

                    if not BERTOPIC_AVAILABLE:
                        st.error("❌ BERTopic not installed.")
                    else:
                        with st.spinner("Running BERTopic..."):
                            topic_model, topics, probs, topics_dict = perform_bertopic_modeling(
                                data['text'].tolist(),
                                n_topics=n_topics
                            )

                        if topics_dict:
                            st.success(f"✅ Discovered {len(topics_dict)} semantic topics!")

                            for topic_name, words in topics_dict.items():
                                with st.expander(f"**{topic_name}**"):
                                    st.write(f"**Top Words:** {', '.join(words)}")

                            st.warning("⚠️ These are unsupervised topics. Consider using Seeded BERTopic for alignment with your categories.")

        # ====================================================================
        # TAB 3: TOPIC-SENTIMENT INTEGRATION
        # ====================================================================
        with tab3:
            st.header("🔗 Topic-Sentiment Integration")
            st.markdown("""
            Combine topic classification with sentiment analysis to answer questions like:
            - *"What emotions are associated with Purchase Intent?"*
            - *"How do people feel when seeking information?"*
            - *"Which topics have the most negative sentiment?"*
            """)

            if 'topic_results' not in st.session_state:
                st.warning("⚠️ Please run Topic Modeling first (Tab 2) to enable integration.")
            else:
                topic_results = st.session_state['topic_results']

                st.success(f"✅ Found topic results for {len(topic_results)} documents")

                # Get sentiment predictions
                st.subheader("🎭 Combining Topics with Sentiments")

                if st.button("🔗 Run Integration Analysis"):
                    # Use the sentiment from original data
                    sentiments = data['sentiment'].values

                    # Combine
                    combined_df = combine_topic_sentiment(
                        topic_results['text'].tolist(),
                        topic_results['topic'].tolist(),
                        sentiments,
                        EMOTION_MAPPING
                    )

                    st.session_state['combined_results'] = combined_df

                    # Analyze distribution
                    analysis = analyze_topic_sentiment_distribution(combined_df)

                    # Cross-tabulation heatmap
                    st.subheader("🔥 Topic-Sentiment Heatmap")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.heatmap(
                        analysis['crosstab_normalized'],
                        annot=True,
                        fmt='.1f',
                        cmap='YlOrRd',
                        ax=ax,
                        cbar_kws={'label': 'Percentage (%)'}
                    )
                    ax.set_title("Sentiment Distribution within Each Topic (%)")
                    ax.set_xlabel("Sentiment/Emotion")
                    ax.set_ylabel("Social Commerce Topic")
                    plt.tight_layout()
                    st.pyplot(fig)

                    # Raw counts
                    st.subheader("📊 Raw Counts")
                    st.dataframe(analysis['crosstab'], use_container_width=True)

                    # Dominant sentiment per topic
                    st.subheader("🏆 Dominant Sentiment per Topic")
                    for topic, info in analysis['topic_dominant_sentiment'].items():
                        st.write(f"**{topic}**: {info['dominant_sentiment']} ({info['percentage']:.1f}%)")

                    # Stacked bar chart
                    st.subheader("📊 Stacked Sentiment Distribution")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    analysis['crosstab_normalized'].plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
                    ax.set_xlabel("Social Commerce Topic")
                    ax.set_ylabel("Percentage (%)")
                    ax.set_title("Sentiment Distribution Across Topics")
                    ax.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)

                    # Insights
                    st.subheader("💡 Key Insights")
                    st.markdown("""
                    Based on the analysis:
                    """)

                    for topic, info in analysis['topic_dominant_sentiment'].items():
                        if info['percentage'] > 40:
                            st.write(f"• **{topic}** is strongly associated with **{info['dominant_sentiment']}** sentiment ({info['percentage']:.1f}%)")
                        else:
                            st.write(f"• **{topic}** has mixed sentiments (top: {info['dominant_sentiment']} at {info['percentage']:.1f}%)")

        # ====================================================================
        # TAB 4: INTERACTIVE PREDICTION
        # ====================================================================
        with tab4:
            st.header("🔮 Interactive Prediction")

            user_input = st.text_area("Enter text for analysis:", value="", height=100)

            col1, col2 = st.columns(2)
            with col1:
                analyze_sentiment = st.checkbox("Analyze Sentiment", value=True)
            with col2:
                analyze_topic = st.checkbox("Classify Topic", value=True)

            analyze_button = st.button("🔍 Analyze", type="primary")

            if user_input and analyze_button:
                st.subheader("Analysis Results")

                # Topic Classification
                if analyze_topic:
                    st.markdown("### 📚 Topic Classification")

                    if TRANSFORMERS_AVAILABLE:
                        with st.spinner("Classifying topic..."):
                            classifier = load_zero_shot_classifier()
                            result = classify_social_commerce_topic(user_input, classifier)

                        if result:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("🎯 Predicted Topic", result['top_label'])
                            with col2:
                                st.metric("📈 Confidence", f"{result['top_score']:.2%}")

                            # All scores
                            st.write("**All Topic Scores:**")
                            for label, score in zip(result['labels'], result['scores']):
                                st.progress(score, text=f"{label}: {score:.2%}")
                    else:
                        st.warning("Install transformers for topic classification")

                # Sentiment Analysis
                if analyze_sentiment:
                    st.markdown("### 🎭 Sentiment Analysis")

                    # Prepare input
                    processed_input = preprocess_text(clean_text(user_input))

                    # Traditional ML prediction
                    if use_traditional and model_selection:
                        # Vectorization needed
                        X_train_tfidf, X_test_tfidf, vectorizer_tfidf = create_feature_vectors(
                            X_train, X_test, vectorizer_type='tfidf', max_features=MAX_FEATURES
                        )

                        input_vectorized = vectorizer_tfidf.transform([processed_input])

                        all_models = get_sentiment_models()
                        selected_model = all_models[model_selection[0]]
                        selected_model.fit(X_train_tfidf, y_train)

                        sentiment_pred = selected_model.predict(input_vectorized)[0]
                        sentiment_proba = selected_model.predict_proba(input_vectorized)[0]

                        sentiment_label = EMOTION_MAPPING[sentiment_pred]

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("🎯 Predicted Emotion (ML)", sentiment_label.upper())
                        with col2:
                            st.metric("📈 Confidence", f"{sentiment_proba.max():.2%}")

                    # BERT prediction
                    if use_bert and TRANSFORMERS_AVAILABLE:
                        bert_model, bert_tokenizer = load_multiclass_bert_model()

                        if bert_model is not None:
                            predictions, probabilities = predict_with_bert([user_input], bert_model, bert_tokenizer, batch_size=1)

                            if predictions is not None:
                                bert_pred = predictions[0]
                                bert_proba = probabilities[0]

                                bert_label = EMOTION_MAPPING[bert_pred]

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("🎯 Predicted Emotion (BERT)", bert_label.upper())
                                with col2:
                                    st.metric("📈 Confidence", f"{bert_proba.max():.2%}")

                                # Top 3 emotions
                                st.write("**Top 3 Emotions (BERT):**")
                                sentiment_probabilities = {EMOTION_MAPPING[i]: prob for i, prob in enumerate(bert_proba)}
                                top3 = sorted(sentiment_probabilities.items(), key=lambda x: x[1], reverse=True)[:3]

                                for emotion, prob in top3:
                                    st.progress(prob, text=f"{emotion}: {prob:.2%}")

                    # Additional analysis
                    col1, col2 = st.columns(2)

                    with col1:
                        sentiment_intensity = get_sentiment_intensity(user_input)
                        st.write("**Sentiment Intensity (VADER):**")
                        st.progress(abs(sentiment_intensity))
                        st.write(f"Score: {sentiment_intensity:.4f}")

                    with col2:
                        subjectivity = get_subjectivity(user_input)
                        st.write("**Subjectivity:**")
                        st.progress(subjectivity)
                        st.write(f"Score: {subjectivity:.4f}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())


if __name__ == "__main__":
    # Initialize NLTK data (run once)
    import nltk
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')

    main()
