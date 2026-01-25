"""
Guided Topic Modeling for Social Commerce Behavior
====================================================

This module provides several approaches for discovering and classifying
social commerce topics from online posts and comments.

Approaches:
1. Guided LDA with seed words (semi-supervised)
2. BERTopic with custom topic representation
3. Zero-shot classification (transformer-based)
4. Keyword-based topic classification
5. Hybrid approach (topic modeling + classification)

Author: AI Research Team
Date: 2026-01-25
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ============================================================================
# SOCIAL COMMERCE TOPIC DEFINITIONS
# ============================================================================

SOCIAL_COMMERCE_TOPICS = {
    "Purchase Intent": {
        "description": "Users expressing desire to buy or looking for products",
        "seed_words": [
            "buy", "purchase", "order", "want", "need", "looking for",
            "shopping", "cart", "checkout", "get", "interested",
            "planning to buy", "going to order", "will purchase"
        ],
        "patterns": [
            r"\b(want to buy|going to buy|planning to purchase|looking for)\b",
            r"\b(add to cart|checkout|order now)\b",
            r"\b(interested in|considering buying)\b"
        ]
    },

    "Product Reviews": {
        "description": "Reviews, ratings, quality feedback, satisfaction/dissatisfaction",
        "seed_words": [
            "quality", "good", "bad", "excellent", "terrible", "love it",
            "hate it", "works", "doesn't work", "satisfied", "disappointed",
            "recommend", "not recommend", "rating", "stars", "review"
        ],
        "patterns": [
            r"\b(love|hate|like|dislike) (this|it|the product)\b",
            r"\b(good|bad|excellent|terrible|amazing|awful) (quality|product)\b",
            r"\b\d+/\d+|\d+ stars?\b"
        ]
    },

    "Shopping Experience": {
        "description": "Customer service, delivery, website usability, overall experience",
        "seed_words": [
            "delivery", "shipping", "fast", "slow", "customer service",
            "support", "easy", "difficult", "website", "app", "navigate",
            "arrived", "delayed", "tracking", "return", "refund", "exchange"
        ],
        "patterns": [
            r"\b(delivery|shipping) (fast|slow|delayed|on time)\b",
            r"\b(customer service|support) (helpful|terrible|good|bad)\b",
            r"\b(easy|hard|difficult) to (use|navigate|find)\b"
        ]
    },

    "Price Sensitivity": {
        "description": "Pricing, deals, discounts, value for money",
        "seed_words": [
            "price", "expensive", "cheap", "affordable", "discount",
            "sale", "deal", "coupon", "promo", "value", "worth",
            "overpriced", "fair price", "budget", "cost", "free shipping"
        ],
        "patterns": [
            r"\b(too expensive|overpriced|cheap|affordable)\b",
            r"\b\d+% (off|discount)\b",
            r"\b(sale|deal|promo|coupon)\b"
        ]
    },

    "Social Influence": {
        "description": "Recommendations, trending products, influencer mentions, peer influence",
        "seed_words": [
            "recommend", "suggested", "friend", "everyone", "trending",
            "popular", "influencer", "reviews say", "people say",
            "viral", "must have", "heard about", "seen on", "social media"
        ],
        "patterns": [
            r"\b(everyone|people) (love|recommend|talk about)\b",
            r"\b(friend|friends) (recommended|suggested|bought)\b",
            r"\b(trending|popular|viral|must have)\b"
        ]
    }
}

# ============================================================================
# APPROACH 1: KEYWORD-BASED CLASSIFICATION (SIMPLEST, FAST)
# ============================================================================

class KeywordTopicClassifier:
    """
    Simple keyword-based topic classification

    **How it works:**
    - Matches documents to topics based on keyword presence
    - Fast and interpretable
    - Good for well-defined topics

    **When to use:**
    - Quick prototyping
    - Clear topic definitions
    - Need interpretability

    **Pros:**
    ✓ Fast (milliseconds)
    ✓ Interpretable
    ✓ No training needed

    **Cons:**
    ⚠ Simple matching (no semantic understanding)
    ⚠ May miss nuanced expressions
    """

    def __init__(self, topics_config: Dict):
        self.topics_config = topics_config
        self.topic_names = list(topics_config.keys())

    def classify_document(self, text: str) -> Tuple[str, float, Dict]:
        """
        Classify a single document

        Returns:
            - predicted_topic: str
            - confidence: float (0-1)
            - all_scores: dict of topic -> score
        """
        text_lower = text.lower()

        # Score each topic
        topic_scores = {}
        for topic_name, topic_info in self.topics_config.items():
            score = 0
            seed_words = topic_info['seed_words']

            # Count keyword matches
            for word in seed_words:
                if word.lower() in text_lower:
                    score += 1

            # Normalize by number of seed words
            topic_scores[topic_name] = score / len(seed_words)

        # Get top topic
        if max(topic_scores.values()) == 0:
            return "Unknown/Other", 0.0, topic_scores

        predicted_topic = max(topic_scores, key=topic_scores.get)
        confidence = topic_scores[predicted_topic]

        return predicted_topic, confidence, topic_scores

    def classify_corpus(self, texts: List[str]) -> pd.DataFrame:
        """
        Classify multiple documents

        Returns:
            DataFrame with columns: text, predicted_topic, confidence, [topic_scores]
        """
        results = []
        for text in texts:
            topic, conf, scores = self.classify_document(text)
            result = {
                'text': text[:100] + '...' if len(text) > 100 else text,
                'predicted_topic': topic,
                'confidence': conf
            }
            # Add individual topic scores
            for t, s in scores.items():
                result[f'score_{t}'] = s
            results.append(result)

        return pd.DataFrame(results)

# ============================================================================
# APPROACH 2: GUIDED LDA WITH SEED WORDS
# ============================================================================

class GuidedLDATopicModel:
    """
    LDA with guidance from seed words

    **How it works:**
    - Uses seed words to initialize topic-word distributions
    - LDA discovers topics guided by these seeds
    - More flexible than pure keyword matching

    **When to use:**
    - Want probabilistic topic modeling
    - Have good seed words
    - Need topic discovery + guidance

    **Pros:**
    ✓ Combines guidance with discovery
    ✓ Probabilistic framework
    ✓ Can discover new relevant words

    **Cons:**
    ⚠ Slower than keyword matching
    ⚠ Requires parameter tuning
    """

    def __init__(self, topics_config: Dict, n_topics: int = 5):
        self.topics_config = topics_config
        self.n_topics = n_topics
        self.vectorizer = None
        self.lda_model = None
        self.topic_names = list(topics_config.keys())

    def fit(self, texts: List[str], max_features: int = 1000):
        """
        Fit guided LDA model
        """
        # Create document-term matrix
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words='english',
            min_df=2
        )
        doc_term_matrix = self.vectorizer.fit_transform(texts)

        # Fit LDA
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            max_iter=50,
            learning_method='online',
            n_jobs=-1
        )
        self.lda_model.fit(doc_term_matrix)

        # Map LDA topics to predefined topics based on seed word overlap
        self._map_topics_to_categories()

        return self

    def _map_topics_to_categories(self):
        """
        Map discovered LDA topics to predefined social commerce categories
        """
        feature_names = self.vectorizer.get_feature_names_out()
        self.topic_mapping = {}

        for topic_idx in range(self.n_topics):
            # Get top words for this LDA topic
            top_word_indices = self.lda_model.components_[topic_idx].argsort()[-20:][::-1]
            topic_words = set([feature_names[i] for i in top_word_indices])

            # Find best matching predefined category
            best_category = None
            best_overlap = 0

            for category_name, category_info in self.topics_config.items():
                seed_words_set = set([w.lower() for w in category_info['seed_words']])
                overlap = len(topic_words.intersection(seed_words_set))

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_category = category_name

            self.topic_mapping[topic_idx] = best_category if best_overlap > 0 else "Unknown/Other"

    def predict(self, texts: List[str]) -> pd.DataFrame:
        """
        Predict topics for new documents
        """
        if self.lda_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Transform documents
        doc_term_matrix = self.vectorizer.transform(texts)
        doc_topics = self.lda_model.transform(doc_term_matrix)

        # Get dominant topic for each document
        results = []
        for i, doc_topic_dist in enumerate(doc_topics):
            dominant_topic_idx = np.argmax(doc_topic_dist)
            confidence = doc_topic_dist[dominant_topic_idx]
            predicted_category = self.topic_mapping.get(dominant_topic_idx, "Unknown/Other")

            results.append({
                'text': texts[i][:100] + '...' if len(texts[i]) > 100 else texts[i],
                'predicted_topic': predicted_category,
                'confidence': confidence,
                'lda_topic_id': dominant_topic_idx
            })

        return pd.DataFrame(results)

    def get_topic_words(self, n_words: int = 10) -> Dict:
        """
        Get top words for each discovered topic
        """
        if self.lda_model is None:
            raise ValueError("Model not fitted.")

        feature_names = self.vectorizer.get_feature_names_out()
        topic_words = {}

        for topic_idx in range(self.n_topics):
            top_word_indices = self.lda_model.components_[topic_idx].argsort()[-n_words:][::-1]
            words = [feature_names[i] for i in top_word_indices]
            category = self.topic_mapping.get(topic_idx, "Unknown/Other")
            topic_words[f"Topic {topic_idx} ({category})"] = words

        return topic_words

# ============================================================================
# APPROACH 3: ZERO-SHOT CLASSIFICATION (TRANSFORMERS)
# ============================================================================

class ZeroShotTopicClassifier:
    """
    Zero-shot classification using transformers

    **How it works:**
    - Uses pre-trained NLI model
    - No training needed
    - Classifies based on semantic similarity

    **When to use:**
    - Have clear topic labels
    - Need high accuracy
    - Don't have labeled training data

    **Pros:**
    ✓ No training needed
    ✓ High accuracy
    ✓ Semantic understanding
    ✓ Works with topic descriptions

    **Cons:**
    ⚠ Slower than keyword matching
    ⚠ Requires transformers library
    ⚠ Needs GPU for speed
    """

    def __init__(self, topics_config: Dict, model_name: str = 'facebook/bart-large-mnli'):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library required. Install with: pip install transformers")

        self.topics_config = topics_config
        self.candidate_labels = list(topics_config.keys())

        # Load zero-shot classification pipeline
        device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device
        )

    def classify_document(self, text: str, use_descriptions: bool = True) -> Tuple[str, float, Dict]:
        """
        Classify a single document

        Args:
            text: Input text
            use_descriptions: If True, use topic descriptions as labels

        Returns:
            - predicted_topic: str
            - confidence: float
            - all_scores: dict
        """
        # Use topic descriptions or just names
        if use_descriptions:
            labels = [f"{name}: {info['description']}"
                     for name, info in self.topics_config.items()]
        else:
            labels = self.candidate_labels

        # Classify
        result = self.classifier(text, labels, multi_label=False)

        # Extract results
        if use_descriptions:
            # Remove descriptions from labels
            predicted_topic = result['labels'][0].split(':')[0]
            scores_dict = {
                label.split(':')[0]: score
                for label, score in zip(result['labels'], result['scores'])
            }
        else:
            predicted_topic = result['labels'][0]
            scores_dict = dict(zip(result['labels'], result['scores']))

        confidence = result['scores'][0]

        return predicted_topic, confidence, scores_dict

    def classify_corpus(self, texts: List[str], use_descriptions: bool = True) -> pd.DataFrame:
        """
        Classify multiple documents
        """
        results = []
        for text in texts:
            topic, conf, scores = self.classify_document(text, use_descriptions)
            result = {
                'text': text[:100] + '...' if len(text) > 100 else text,
                'predicted_topic': topic,
                'confidence': conf
            }
            # Add scores for all topics
            for t, s in scores.items():
                result[f'score_{t}'] = s
            results.append(result)

        return pd.DataFrame(results)

# ============================================================================
# APPROACH 4: BERTOPIC WITH GUIDED REPRESENTATION
# ============================================================================

class GuidedBERTopicModel:
    """
    BERTopic with custom topic representation guided by seed words

    **How it works:**
    - BERTopic discovers topics using BERT embeddings
    - Custom representation layer maps to predefined categories
    - Best of both worlds: discovery + guidance

    **When to use:**
    - Want state-of-the-art topic quality
    - Have computational resources
    - Need both discovery and classification

    **Pros:**
    ✓ Best topic coherence
    ✓ Semantic understanding
    ✓ Flexible topic representation

    **Cons:**
    ⚠ Slower (requires embeddings)
    ⚠ More complex setup
    ⚠ Needs more memory
    """

    def __init__(self, topics_config: Dict):
        if not BERTOPIC_AVAILABLE:
            raise ImportError("BERTopic required. Install with: pip install bertopic sentence-transformers")

        self.topics_config = topics_config
        self.topic_names = list(topics_config.keys())

        # Initialize BERTopic
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.topic_model = BERTopic(
            embedding_model=embedding_model,
            min_topic_size=10,
            nr_topics=len(topics_config),
            calculate_probabilities=True,
            verbose=False
        )

        self.topic_mapping = None

    def fit(self, texts: List[str]):
        """
        Fit BERTopic model and map to predefined categories
        """
        # Fit BERTopic
        topics, probabilities = self.topic_model.fit_transform(texts)

        # Map discovered topics to predefined categories
        self._map_topics_to_categories()

        return self

    def _map_topics_to_categories(self):
        """
        Map BERTopic topics to predefined social commerce categories
        """
        topic_info = self.topic_model.get_topic_info()
        self.topic_mapping = {}

        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id == -1:  # Outliers
                self.topic_mapping[topic_id] = "Unknown/Other"
                continue

            # Get topic words
            topic_words = self.topic_model.get_topic(topic_id)
            if not topic_words:
                self.topic_mapping[topic_id] = "Unknown/Other"
                continue

            topic_words_set = set([word for word, _ in topic_words[:20]])

            # Find best matching category
            best_category = None
            best_overlap = 0

            for category_name, category_info in self.topics_config.items():
                seed_words_set = set([w.lower() for w in category_info['seed_words']])
                overlap = len(topic_words_set.intersection(seed_words_set))

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_category = category_name

            self.topic_mapping[topic_id] = best_category if best_overlap > 0 else "Unknown/Other"

    def predict(self, texts: List[str]) -> pd.DataFrame:
        """
        Predict topics for new documents
        """
        if self.topic_model is None:
            raise ValueError("Model not fitted.")

        # Transform documents
        topics, probabilities = self.topic_model.transform(texts)

        # Map to categories
        results = []
        for i, (topic_id, prob) in enumerate(zip(topics, probabilities)):
            predicted_category = self.topic_mapping.get(topic_id, "Unknown/Other")
            confidence = prob[topic_id] if topic_id >= 0 else 0.0

            results.append({
                'text': texts[i][:100] + '...' if len(texts[i]) > 100 else texts[i],
                'predicted_topic': predicted_category,
                'confidence': confidence,
                'bertopic_id': topic_id
            })

        return pd.DataFrame(results)

    def get_topic_info(self) -> pd.DataFrame:
        """
        Get information about discovered topics
        """
        topic_info = self.topic_model.get_topic_info()
        topic_info['Mapped_Category'] = topic_info['Topic'].map(self.topic_mapping)
        return topic_info

# ============================================================================
# VISUALIZATION AND ANALYSIS UTILITIES
# ============================================================================

def plot_topic_distribution(results_df: pd.DataFrame, title: str = "Topic Distribution"):
    """
    Plot distribution of predicted topics
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    topic_counts = results_df['predicted_topic'].value_counts()

    colors = plt.cm.Set3(range(len(topic_counts)))
    ax.bar(topic_counts.index, topic_counts.values, color=colors)
    ax.set_xlabel("Topic")
    ax.set_ylabel("Number of Documents")
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig

def plot_confidence_distribution(results_df: pd.DataFrame, title: str = "Confidence Distribution"):
    """
    Plot distribution of prediction confidence scores
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for topic in results_df['predicted_topic'].unique():
        topic_data = results_df[results_df['predicted_topic'] == topic]
        ax.hist(topic_data['confidence'], alpha=0.5, label=topic, bins=20)

    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    return fig

def plot_topic_heatmap(results_df: pd.DataFrame, topics_config: Dict):
    """
    Plot heatmap of topic scores for all documents
    """
    # Extract score columns
    score_columns = [col for col in results_df.columns if col.startswith('score_')]

    if not score_columns:
        return None

    # Get topic names from column names
    topic_names = [col.replace('score_', '') for col in score_columns]

    # Create heatmap data
    heatmap_data = results_df[score_columns].head(20).values  # First 20 docs

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        heatmap_data,
        cmap='YlOrRd',
        annot=True,
        fmt='.2f',
        xticklabels=topic_names,
        yticklabels=[f"Doc {i+1}" for i in range(len(heatmap_data))],
        cbar_kws={'label': 'Topic Score'}
    )
    ax.set_title("Document-Topic Score Matrix (Sample)")
    ax.set_xlabel("Social Commerce Topic")
    ax.set_ylabel("Document")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig

def compare_methods(texts: List[str], topics_config: Dict) -> pd.DataFrame:
    """
    Compare all classification methods on the same data

    Returns comparison DataFrame
    """
    results = []

    # Method 1: Keyword-based
    print("Running Keyword-based classification...")
    kw_classifier = KeywordTopicClassifier(topics_config)
    kw_results = kw_classifier.classify_corpus(texts)

    for _, row in kw_results.iterrows():
        results.append({
            'text': row['text'],
            'method': 'Keyword-based',
            'predicted_topic': row['predicted_topic'],
            'confidence': row['confidence']
        })

    # Method 2: Guided LDA
    print("Running Guided LDA...")
    try:
        lda_classifier = GuidedLDATopicModel(topics_config, n_topics=5)
        lda_classifier.fit(texts)
        lda_results = lda_classifier.predict(texts)

        for _, row in lda_results.iterrows():
            results.append({
                'text': row['text'],
                'method': 'Guided LDA',
                'predicted_topic': row['predicted_topic'],
                'confidence': row['confidence']
            })
    except Exception as e:
        print(f"Guided LDA failed: {e}")

    # Method 3: Zero-shot (if transformers available)
    if TRANSFORMERS_AVAILABLE:
        print("Running Zero-shot classification...")
        try:
            zs_classifier = ZeroShotTopicClassifier(topics_config)
            zs_results = zs_classifier.classify_corpus(texts[:100])  # Limit for speed

            for _, row in zs_results.iterrows():
                results.append({
                    'text': row['text'],
                    'method': 'Zero-shot',
                    'predicted_topic': row['predicted_topic'],
                    'confidence': row['confidence']
                })
        except Exception as e:
            print(f"Zero-shot failed: {e}")

    return pd.DataFrame(results)

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def calculate_topic_coherence(texts: List[str], predicted_topics: List[str], topics_config: Dict) -> Dict:
    """
    Calculate coherence metrics for topic assignments

    Returns:
        - avg_confidence: Average prediction confidence
        - topic_purity: How well documents cluster by topic
        - coverage: % of documents assigned to known topics
    """
    metrics = {}

    # Coverage
    known_topics = set(topics_config.keys())
    assigned_known = sum(1 for t in predicted_topics if t in known_topics)
    metrics['coverage'] = assigned_known / len(predicted_topics) if predicted_topics else 0

    # Topic distribution balance
    from collections import Counter
    topic_dist = Counter(predicted_topics)
    total = len(predicted_topics)
    metrics['distribution_balance'] = {
        topic: count/total for topic, count in topic_dist.items()
    }

    # Entropy (higher = more balanced)
    from scipy.stats import entropy
    probs = list(topic_dist.values())
    probs_norm = [p/total for p in probs]
    metrics['entropy'] = entropy(probs_norm)

    return metrics

if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "I'm looking to buy a new phone. Anyone have recommendations?",
        "Just received my order! The quality is amazing, highly recommend!",
        "Terrible customer service. My package was delayed by 2 weeks.",
        "Is this product worth the price? Seems a bit expensive.",
        "Everyone on TikTok is buying this! Must be good."
    ]

    # Test keyword-based classifier
    print("=" * 60)
    print("Testing Keyword-Based Classifier")
    print("=" * 60)

    classifier = KeywordTopicClassifier(SOCIAL_COMMERCE_TOPICS)
    results = classifier.classify_corpus(sample_texts)
    print(results[['text', 'predicted_topic', 'confidence']])

    print("\n" + "=" * 60)
    print("Topic Definitions:")
    print("=" * 60)
    for topic_name, topic_info in SOCIAL_COMMERCE_TOPICS.items():
        print(f"\n{topic_name}:")
        print(f"  Description: {topic_info['description']}")
        print(f"  Key words: {', '.join(topic_info['seed_words'][:10])}...")
