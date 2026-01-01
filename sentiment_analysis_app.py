"""
Sentiment Analysis and Topic Modeling Application - Enhanced with BERT & BERTopic
==================================================================================

This application provides state-of-the-art text analysis capabilities including:
1. Sentiment Analysis using traditional ML + BERT transformers
2. Topic Modeling using LDA, NMF, and BERTopic
3. Interactive Streamlit interface for real-time predictions

Author: AI Research Team
Date: 2026-01-01
Version: 2.0 (Enhanced with Transformers)
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.decomposition import LatentDirichletAllocation, NMF
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from scipy.stats import entropy
import streamlit as st
import matplotlib.pyplot as plt
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
    st.warning("⚠️ Transformers not installed. BERT models will not be available. Install with: pip install transformers torch")

try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    st.warning("⚠️ BERTopic not installed. Advanced topic modeling not available. Install with: pip install bertopic sentence-transformers")

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
# DATA LOADING AND PREPROCESSING
# ============================================================================

@st.cache_data
def load_and_preprocess_data(file_path):
    """
    Load and preprocess the dataset.

    **Functionality:**
    - Loads CSV data
    - Drops ID column and duplicates
    - Samples data for faster processing
    - Renames columns for convenience

    **Evaluation:**
    ✓ Good: Using drop_duplicates() to ensure unique samples
    ✓ Good: Sampling strategy for faster experimentation
    ⚠ Consider: Making sample_fraction configurable via UI

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Preprocessed dataframe
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

    **Functionality:**
    - Converts to lowercase
    - Removes URLs
    - Removes digits
    - Removes punctuation

    **Evaluation:**
    ✓ Good: Comprehensive cleaning pipeline
    ✓ Good: Handles non-string inputs
    ⚠ Consider: Preserving emojis for sentiment analysis
    ⚠ Consider: Handling contractions (don't -> do not)

    Args:
        text (str): Raw text to clean

    Returns:
        str: Cleaned text
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

    **Functionality:**
    - Tokenizes text into words
    - Removes stopwords
    - Lemmatizes words to their root form

    **Evaluation:**
    ✓ Good: Using lemmatization instead of stemming (preserves meaning better)
    ✓ Good: Stopword removal reduces noise
    ⚠ Consider: Making stopword list customizable (keep negation words like 'not')
    ⚠ Consider: Using spaCy for lemmatization (faster and more accurate)

    Args:
        text (str): Cleaned text

    Returns:
        str: Preprocessed text
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

    **Functionality:**
    - TF-IDF: Weighs terms by frequency and rarity (default)
    - CountVectorizer: Simple word counts

    **Evaluation:**
    ✓ Good: TF-IDF is excellent for sentiment analysis
    ✓ Good: Configurable vectorizer type
    ⚠ Consider: Adding n-gram features (bigrams, trigrams) for better context
    ⚠ Consider: Using Word2Vec, GloVe, or BERT embeddings for deep learning models

    Args:
        X_train: Training text data
        X_test: Test text data
        vectorizer_type (str): 'tfidf' or 'count'
        max_features (int): Maximum number of features

    Returns:
        tuple: (X_train_vectorized, X_test_vectorized, vectorizer)
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

    **Models Included:**

    1. **Logistic Regression**
       - Functionality: Linear model for classification
       - Evaluation: ✓ Excellent baseline, fast, interpretable
       - Best for: Quick prototyping, interpretable results

    2. **Random Forest**
       - Functionality: Ensemble of decision trees
       - Evaluation: ✓ Good for non-linear patterns, handles overfitting well
       - Best for: Complex patterns, feature importance analysis

    3. **Naive Bayes (MultinomialNB)**
       - Functionality: Probabilistic classifier based on Bayes theorem
       - Evaluation: ✓ Very fast, works well with text data
       - Best for: Large datasets, real-time predictions

    4. **Support Vector Machine (SVM)**
       - Functionality: Finds optimal hyperplane for classification
       - Evaluation: ✓ Excellent for high-dimensional text data
       - ⚠ Warning: Can be slow on large datasets
       - Best for: Small to medium datasets with high accuracy requirements

    5. **Gradient Boosting**
       - Functionality: Sequential ensemble learning
       - Evaluation: ✓ Often achieves highest accuracy among traditional methods
       - ⚠ Warning: Slower training, prone to overfitting
       - Best for: Competitions, maximum accuracy needed

    Returns:
        dict: Dictionary of model names and instances
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
# BERT-BASED SENTIMENT ANALYSIS (STATE-OF-THE-ART)
# ============================================================================

@st.cache_resource
def load_bert_sentiment_model():
    """
    Load pre-trained BERT model for sentiment analysis.

    **Model: DistilBERT**
    - Functionality: Distilled version of BERT (40% smaller, 60% faster)
    - Pre-trained on: Stanford Sentiment Treebank (SST-2)
    - Output: Binary sentiment (positive/negative) with confidence scores

    **Evaluation:**
    ✓ Excellent: State-of-the-art accuracy (>90% on SST-2)
    ✓ Good: Faster than full BERT while maintaining performance
    ✓ Good: Pre-trained, no fine-tuning needed for basic sentiment
    ⚠ Limitation: Binary sentiment (can be adapted for multi-class)

    **How well is it used:**
    ✓ Perfect for: General sentiment analysis with high accuracy
    ✓ Production-ready: Optimized for speed and memory
    ⚠ Consider: Fine-tuning on your specific domain for better results

    **Alternative Models:**
    - 'nlptown/bert-base-multilingual-uncased-sentiment' (5-class sentiment)
    - 'cardiffnlp/twitter-roberta-base-sentiment' (Twitter-specific)
    - 'distilbert-base-uncased' (for fine-tuning on custom data)

    Returns:
        pipeline: Hugging Face sentiment analysis pipeline
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

    **Model: DistilBERT fine-tuned on emotions**
    - Functionality: Classifies text into 6 emotions (joy, sadness, anger, fear, love, surprise)
    - Architecture: DistilBERT + classification head
    - Training: Fine-tuned on emotion datasets

    **Evaluation:**
    ✓ Excellent: Matches emotion classes in your dataset
    ✓ Good: Context-aware emotion detection
    ✓ Good: Handles nuanced emotional expressions

    **How well is it used:**
    ✓ Perfect for: Emotion classification tasks
    ✓ Better than: Traditional ML for complex emotional context
    ⚠ Slower than: Traditional ML (trade-off for accuracy)

    Returns:
        tuple: (model, tokenizer)
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

    **Functionality:**
    - Tokenizes input texts
    - Runs inference through BERT model
    - Returns emotion predictions and probabilities

    **Evaluation:**
    ✓ Good: Batch processing for efficiency
    ✓ Good: Handles variable-length inputs
    ✓ Good: Returns both predictions and confidence scores

    Args:
        texts (list): List of text strings
        model: BERT model
        tokenizer: BERT tokenizer
        batch_size (int): Batch size for inference

    Returns:
        tuple: (predictions, probabilities)
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
# MODEL EVALUATION
# ============================================================================

def cross_validate_model(model, X_train_tfidf, y_train, cv=5, scoring='accuracy'):
    """
    Perform cross-validation on a model.

    **Functionality:**
    - Uses StratifiedKFold to maintain class distribution
    - Evaluates model on multiple folds
    - Returns mean and standard deviation of scores

    **Evaluation:**
    ✓ Good: Stratified K-Fold maintains class balance
    ✓ Good: Multiple folds reduce variance
    ⚠ Consider: Using additional scoring metrics (F1, ROC-AUC)

    Args:
        model: Scikit-learn model instance
        X_train_tfidf: Vectorized training features
        y_train: Training labels
        cv (int): Number of folds
        scoring (str): Scoring metric

    Returns:
        tuple: (mean_score, std_score)
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train_tfidf, y_train, cv=skf, scoring=scoring, n_jobs=-1)
    return scores.mean(), scores.std()


def evaluate_model_optimized(model, X_train_tfidf, X_test_tfidf, y_train, y_test, all_classes, calc_auc=False):
    """
    Comprehensive model evaluation with multiple metrics.

    **Metrics Calculated:**
    - Accuracy: Overall correctness
    - Precision: How many predicted positives are correct
    - Recall: How many actual positives were found
    - F1 Score: Harmonic mean of precision and recall
    - ROC-AUC: Area under ROC curve (if calc_auc=True)

    **Evaluation:**
    ✓ Good: Comprehensive metrics suite
    ✓ Good: Handles multi-class classification
    ✓ Good: Aligns probabilities for classes not seen in training
    ⚠ Consider: Adding confusion matrix
    ⚠ Consider: Per-class metrics for imbalanced datasets

    Args:
        model: Scikit-learn model
        X_train_tfidf: Training features
        X_test_tfidf: Test features
        y_train: Training labels
        y_test: Test labels
        all_classes: All unique class labels
        calc_auc (bool): Whether to calculate ROC-AUC

    Returns:
        tuple: (accuracy, precision, recall, f1, roc_auc)
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

    **Functionality:**
    - Runs BERT inference on test set
    - Calculates standard classification metrics
    - Compares with traditional ML performance

    **Evaluation:**
    ✓ Good: Fair comparison with traditional models
    ✓ Good: Provides comprehensive metrics

    Args:
        bert_model: BERT model
        tokenizer: BERT tokenizer
        X_test: Test texts
        y_test: True labels

    Returns:
        dict: Evaluation metrics
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
# TOPIC MODELING - TRADITIONAL METHODS
# ============================================================================

def perform_topic_modeling(texts, n_topics=5, method='lda', n_top_words=10):
    """
    Perform topic modeling on text data using traditional methods.

    **Methods Available:**

    1. **LDA (Latent Dirichlet Allocation)**
       - Functionality: Probabilistic generative model
       - Evaluation: ✓ Industry standard, interpretable topics
       - Best for: Discovering hidden topics in documents
       - Assumptions: Documents are mixtures of topics

    2. **NMF (Non-Negative Matrix Factorization)**
       - Functionality: Linear algebra factorization
       - Evaluation: ✓ Faster than LDA, clearer topics
       - Best for: Sparse, non-negative data (like text)
       - Advantages: Better for short texts

    **Evaluation:**
    ✓ Good: Supports multiple methods
    ✓ Good: Configurable number of topics
    ⚠ Consider: Adding coherence score calculation
    ⚠ Consider: Adding dynamic topic number selection

    Args:
        texts (list): List of text documents
        n_topics (int): Number of topics to extract
        method (str): 'lda' or 'nmf'
        n_top_words (int): Number of top words per topic

    Returns:
        tuple: (model, vectorizer, feature_names, topics_dict)
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

    **BERTopic Architecture:**
    - Step 1: BERT embeddings (semantic representations)
    - Step 2: UMAP dimensionality reduction
    - Step 3: HDBSCAN clustering
    - Step 4: c-TF-IDF for topic representation

    **Evaluation:**
    ✓ Excellent: State-of-the-art topic modeling
    ✓ Excellent: Captures semantic meaning, not just word co-occurrence
    ✓ Good: Dynamic topic modeling (topics evolve over time)
    ✓ Good: Better topic coherence than LDA/NMF
    ⚠ Requires: More computational resources
    ⚠ Slower than: Traditional methods

    **How well is it used:**
    ✓ Perfect for: Discovering semantic topics in modern text
    ✓ Better than LDA when: You have computational resources and need best quality
    ✓ Production-ready: Can be saved and loaded for inference

    **Advantages over LDA/NMF:**
    1. Semantic understanding (not just word patterns)
    2. Better handling of synonyms and context
    3. More coherent and interpretable topics
    4. Can discover topics of varying granularity

    Args:
        n_topics (int): Target number of topics (auto if None)

    Returns:
        BERTopic: Configured BERTopic model
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
    Perform BERTopic modeling on text data.

    **Functionality:**
    - Creates semantic embeddings using sentence transformers
    - Clusters similar documents
    - Extracts representative keywords for each cluster
    - Returns topics with coherent semantic meaning

    **Evaluation:**
    ✓ Excellent: Superior topic quality compared to LDA/NMF
    ✓ Good: Handles short and long texts equally well
    ✓ Good: Automatically determines optimal number of topics if not specified

    Args:
        texts (list): List of text documents
        n_topics (int): Target number of topics

    Returns:
        tuple: (bertopic_model, topics, probabilities, topic_info)
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

    **Functionality:**
    - Transforms documents into topic space
    - Returns top N topics per document with probabilities

    **Evaluation:**
    ✓ Good: Provides topic distribution per document
    ⚠ Consider: Adding topic visualization

    Args:
        model: Trained topic model (LDA or NMF)
        vectorizer: Fitted vectorizer
        texts (list): List of documents
        top_n (int): Number of top topics to return

    Returns:
        list: List of tuples (topic_idx, probability) for each document
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
# SENTIMENT ANALYSIS UTILITIES
# ============================================================================

def get_sentiment_intensity(text):
    """
    Get sentiment intensity using VADER.

    **Functionality:**
    - Uses VADER (Valence Aware Dictionary and sEntiment Reasoner)
    - Returns compound score [-1, 1]

    **Evaluation:**
    ✓ Good: VADER is excellent for social media text
    ✓ Good: Handles emojis, slang, capitalization
    ⚠ Consider: Combining with TextBlob for robustness

    Args:
        text (str): Input text

    Returns:
        float: Compound sentiment score
    """
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)['compound']


def get_subjectivity(text):
    """
    Get subjectivity score using TextBlob.

    **Functionality:**
    - Measures how subjective (opinionated) vs objective (factual) text is
    - Score range: [0, 1] where 0 is objective, 1 is subjective

    **Evaluation:**
    ✓ Good: TextBlob is simple and effective
    ⚠ Consider: Using pattern library for better accuracy

    Args:
        text (str): Input text

    Returns:
        float: Subjectivity score
    """
    return TextBlob(text).sentiment.subjectivity


def aspect_sentiment_analysis(text):
    """
    Perform aspect-based sentiment analysis using spaCy.

    **Functionality:**
    - Extracts noun phrases (aspects)
    - Calculates sentiment for each aspect

    **Evaluation:**
    ✓ Good: Provides granular sentiment per aspect
    ⚠ Consider: Using dependency parsing for better aspect extraction
    ⚠ Consider: Using ABSA-specific models (PyABSA library)

    Args:
        text (str): Input text

    Returns:
        dict: Aspect -> sentiment score mapping
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

    Args:
        model_name (str): Name of the model
        accuracy (float): Accuracy score
        precision (float): Precision score
        recall (float): Recall score
        f1 (float): F1 score
        roc_auc (float): ROC-AUC score
        cv_mean (float): Cross-validation mean score
        cv_std (float): Cross-validation standard deviation
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
    st.markdown("**Enhanced with BERT & BERTopic** | State-of-the-Art NLP Models")
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
            ['lda', 'nmf', 'bertopic'] if BERTOPIC_AVAILABLE else ['lda', 'nmf']
        )
        n_topics = st.sidebar.slider("Number of Topics", 3, 10, 5)
    else:
        topic_method = 'lda'
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

        # Traditional ML Models
        if use_traditional and model_selection:
            st.markdown("---")
            st.header("🤖 Traditional Machine Learning Models")

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
            st.markdown("---")
            st.header("🚀 BERT-Based Sentiment Analysis")

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

        # Topic Modeling
        if enable_topic_modeling:
            st.markdown("---")
            st.header("📚 Topic Modeling")

            if topic_method == 'bertopic' and BERTOPIC_AVAILABLE:
                st.subheader("🎯 BERTopic - Semantic Topic Discovery")

                with st.spinner("Performing BERTopic analysis... (this may take a minute)"):
                    # Use original text for BERTopic
                    topic_model, topics, probs, topics_dict = perform_bertopic_modeling(
                        data['text'].tolist(),
                        n_topics=n_topics
                    )

                if topics_dict:
                    st.success(f"✓ Discovered {len(topics_dict)} semantic topics!")

                    for topic_name, words in topics_dict.items():
                        with st.expander(f"**{topic_name}**"):
                            st.write(f"**Keywords:** {', '.join(words)}")

                    # Topic distribution
                    if topics is not None:
                        st.subheader("Topic Distribution")
                        fig, ax = plt.subplots(figsize=(10, 4))
                        topic_counts = pd.Series(topics).value_counts().sort_index()
                        topic_counts = topic_counts[topic_counts.index != -1]  # Remove outliers
                        topic_counts.plot(kind='bar', ax=ax, color='green')
                        ax.set_xlabel("Topic ID")
                        ax.set_ylabel("Document Count")
                        ax.set_title("Documents per Topic (BERTopic)")
                        st.pyplot(fig)

            else:
                # Traditional topic modeling
                st.subheader(f"📖 {topic_method.upper()} Topic Modeling")

                with st.spinner(f"Performing {topic_method.upper()} topic modeling..."):
                    topic_model, topic_vectorizer, feature_names, topics_dict = perform_topic_modeling(
                        data['text_processed'].tolist(),
                        n_topics=n_topics,
                        method=topic_method,
                        n_top_words=10
                    )

                st.success(f"✓ Discovered {len(topics_dict)} topics!")

                for topic_name, words in topics_dict.items():
                    with st.expander(f"**{topic_name}**"):
                        st.write(f"**Keywords:** {', '.join(words)}")

        # Interactive prediction
        st.markdown("---")
        st.header("🔮 Interactive Sentiment Prediction")

        user_input = st.text_area("Enter text for sentiment analysis:", value="", height=100)
        analyze_button = st.button("🔍 Analyze Sentiment", type="primary")

        if user_input and analyze_button:
            st.subheader("Analysis Results")

            # Prepare input
            processed_input = preprocess_text(clean_text(user_input))

            # Traditional ML prediction
            if use_traditional and model_selection:
                st.write("### Traditional ML Prediction")

                input_vectorized = vectorizer_tfidf.transform([processed_input])

                selected_model = selected_models[list(selected_models.keys())[0]]
                sentiment_pred = selected_model.predict(input_vectorized)[0]
                sentiment_proba = selected_model.predict_proba(input_vectorized)[0]

                sentiment_label = EMOTION_MAPPING[sentiment_pred]
                st.success(f"**Predicted Emotion (Traditional ML):** {sentiment_label.upper()}")

            # BERT prediction
            if use_bert and TRANSFORMERS_AVAILABLE and bert_model is not None:
                st.write("### BERT Prediction")

                predictions, probabilities = predict_with_bert([user_input], bert_model, bert_tokenizer, batch_size=1)

                if predictions is not None:
                    bert_pred = predictions[0]
                    bert_proba = probabilities[0]

                    bert_label = EMOTION_MAPPING[bert_pred]
                    st.success(f"**Predicted Emotion (BERT):** {bert_label.upper()}")

                    sentiment_probabilities = {
                        EMOTION_MAPPING[i]: prob for i, prob in enumerate(bert_proba)
                    }

                    top3_probabilities = sorted(sentiment_probabilities.items(), key=lambda x: x[1], reverse=True)[:3]

                    # Display top 3 predictions
                    st.write("#### Top 3 Predictions (BERT)")
                    for i, (sentiment, prob) in enumerate(top3_probabilities):
                        color = ['red', 'green', 'blue'][i]
                        st.markdown(
                            f"<div style='text-align:center;'>{sentiment.upper()}: "
                            f"<span style='color:{color}; font-weight:bold;'>{prob * 100:.2f}%</span></div>",
                            unsafe_allow_html=True
                        )

                    # Visualization
                    sentiments, probabilities_vals = zip(*top3_probabilities)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.bar(sentiments, [p * 100 for p in probabilities_vals], color=['red', 'green', 'blue'])
                    ax.set_title("Top 3 Predicted Emotions (BERT)")
                    ax.set_xlabel("Emotion")
                    ax.set_ylabel("Confidence (%)")
                    st.pyplot(fig)

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

            # Aspect-based sentiment
            with st.expander("🔍 Aspect-Based Sentiment Analysis"):
                aspect_sentiments = aspect_sentiment_analysis(user_input)
                if aspect_sentiments:
                    for aspect, sentiment in aspect_sentiments.items():
                        sentiment_emoji = "😊" if sentiment > 0 else "😞" if sentiment < 0 else "😐"
                        st.write(f"{sentiment_emoji} **{aspect}:** {sentiment:.2f}")
                else:
                    st.write("No aspects detected.")

            # Topic assignment
            if enable_topic_modeling and topic_method != 'bertopic':
                with st.expander("📚 Dominant Topics"):
                    doc_topics = get_document_topics(
                        topic_model, topic_vectorizer, [processed_input], top_n=3
                    )

                    for idx, (topic_idx, prob) in enumerate(doc_topics[0]):
                        st.write(f"**Topic {topic_idx + 1}:** {prob:.4f}")
                        st.write(f"Keywords: {', '.join(topics_dict[f'Topic {topic_idx + 1}'])}")

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
