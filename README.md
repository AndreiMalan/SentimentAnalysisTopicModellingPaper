# Sentiment Analysis & Topic Modeling Application

A comprehensive machine learning application for **sentiment analysis** and **topic modeling** with an interactive Streamlit interface.

## 📋 Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Models & Algorithms](#models--algorithms)
- [Topic Modeling](#topic-modeling)
- [Project Structure](#project-structure)
- [Performance Metrics](#performance-metrics)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)

## ✨ Features

### Sentiment Analysis
- **5 Machine Learning Models**: Logistic Regression, Random Forest, Naive Bayes, SVM, Gradient Boosting
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Cross-Validation**: Stratified K-Fold for robust performance estimates
- **Real-time Prediction**: Interactive text input with instant results
- **Advanced Analytics**:
  - Sentiment intensity scoring (VADER)
  - Subjectivity analysis (TextBlob)
  - Aspect-based sentiment analysis (spaCy)
  - Confidence intervals and entropy measures

### Topic Modeling
- **LDA (Latent Dirichlet Allocation)**: Probabilistic topic discovery
- **NMF (Non-Negative Matrix Factorization)**: Linear algebra-based topics
- **Document-Topic Assignment**: Automatic topic labeling for new texts
- **Configurable Topics**: Adjustable number of topics and keywords

### Interactive UI
- **Streamlit Dashboard**: Beautiful, responsive web interface
- **Model Comparison**: Side-by-side performance metrics
- **Visualizations**: Charts, progress bars, and probability distributions
- **File Upload**: Support for custom datasets

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/SentimentAnalysisTopicModellingPaper.git
cd SentimentAnalysisTopicModellingPaper
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download NLTK Data
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('vader_lexicon')"
```

### Step 4: Download spaCy Model
```bash
python -m spacy download en_core_web_sm
```

## 💻 Usage

### Running the Application
```bash
streamlit run sentiment_analysis_app.py
```

The application will open in your browser at `http://localhost:8501`

### Using Your Own Dataset

Your CSV file should have the following format:
```csv
id,text,sentiment
1,"I love this product!",1
2,"This is terrible",0
```

- **text**: The text content to analyze
- **sentiment**: Numeric label (0-5 for emotions: sadness, joy, love, anger, fear, surprise)

Upload your file using the sidebar file uploader.

### Example Code Usage

```python
from sentiment_analysis_app import (
    clean_text,
    preprocess_text,
    get_sentiment_models,
    evaluate_model_optimized
)

# Preprocess text
text = "I absolutely love this product!"
cleaned = clean_text(text)
processed = preprocess_text(cleaned)

# Train a model
models = get_sentiment_models()
lr_model = models['Logistic Regression']
# ... train and evaluate
```

## 🤖 Models & Algorithms

### 1. Logistic Regression
**Functionality**: Linear classification model using sigmoid function

**Evaluation**:
- ✅ **Strengths**: Fast training, interpretable coefficients, good baseline
- ✅ **Best for**: Quick prototyping, interpretable results
- ⚠️ **Limitations**: Assumes linear separability

**When to Use**: First model to try, need interpretability, linear patterns expected

---

### 2. Random Forest
**Functionality**: Ensemble of decision trees with bootstrap aggregating

**Evaluation**:
- ✅ **Strengths**: Handles non-linear patterns, resistant to overfitting, feature importance
- ✅ **Best for**: Complex patterns, feature analysis
- ⚠️ **Limitations**: Slower than linear models, less interpretable

**When to Use**: Non-linear patterns, need feature importance, medium-sized datasets

---

### 3. Naive Bayes (MultinomialNB)
**Functionality**: Probabilistic classifier using Bayes theorem with independence assumption

**Evaluation**:
- ✅ **Strengths**: Very fast, works well with text, handles high dimensions
- ✅ **Best for**: Large datasets, real-time predictions, text classification
- ⚠️ **Limitations**: Independence assumption often violated

**When to Use**: Need speed, large text datasets, baseline model

---

### 4. Support Vector Machine (SVM)
**Functionality**: Finds optimal hyperplane maximizing margin between classes

**Evaluation**:
- ✅ **Strengths**: Excellent for high-dimensional data, effective in text classification
- ✅ **Best for**: Small to medium datasets with high accuracy requirements
- ⚠️ **Limitations**: Slow on large datasets, sensitive to parameter tuning

**When to Use**: Need high accuracy, dataset < 10,000 samples, willing to tune parameters

---

### 5. Gradient Boosting
**Functionality**: Sequential ensemble learning, each tree corrects previous errors

**Evaluation**:
- ✅ **Strengths**: Often highest accuracy, handles complex patterns
- ✅ **Best for**: Competitions, maximum accuracy needed
- ⚠️ **Limitations**: Slower training, prone to overfitting, requires careful tuning

**When to Use**: Need best accuracy, have time for tuning, prevent overfitting

---

### Model Comparison Summary

| Model | Speed | Accuracy | Interpretability | Best Dataset Size |
|-------|-------|----------|------------------|-------------------|
| Logistic Regression | ⚡⚡⚡ | ★★★ | ⭐⭐⭐ | Any |
| Random Forest | ⚡⚡ | ★★★★ | ⭐⭐ | Medium-Large |
| Naive Bayes | ⚡⚡⚡ | ★★★ | ⭐⭐ | Large |
| SVM | ⚡ | ★★★★ | ⭐ | Small-Medium |
| Gradient Boosting | ⚡ | ★★★★★ | ⭐ | Medium |

---

## 🎯 Recommended Models for Future Implementation

### 1. **XGBoost** (Extreme Gradient Boosting)
**Why**: Faster than sklearn's GradientBoosting, built-in regularization, handles missing values

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
```

**Use Case**: Competitions, production systems, imbalanced data

---

### 2. **LightGBM** (Light Gradient Boosting Machine)
**Why**: Very fast, efficient memory usage, handles large datasets

```python
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42
)
```

**Use Case**: Large datasets (>100k samples), production systems

---

### 3. **BERT** (Bidirectional Encoder Representations from Transformers)
**Why**: State-of-the-art for NLP, captures context and semantics

```python
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
```

**Use Case**: Need best accuracy, have GPU, willing to fine-tune

---

### 4. **RoBERTa** (Robustly Optimized BERT)
**Why**: Improved BERT training, better performance

```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=6)
```

**Use Case**: State-of-the-art sentiment analysis, social media text

---

### 5. **CatBoost**
**Why**: Handles categorical features well, minimal hyperparameter tuning

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=5,
    random_state=42,
    verbose=False
)
```

**Use Case**: Mixed categorical/numerical features, quick results

---

## 📚 Topic Modeling

### LDA (Latent Dirichlet Allocation)

**Functionality**:
- Probabilistic generative model
- Assumes documents are mixtures of topics
- Topics are distributions over words

**Evaluation**:
- ✅ **Strengths**: Interpretable, industry standard, probabilistic framework
- ✅ **Best for**: Discovering hidden topics in large document collections
- ⚠️ **Limitations**: Requires tuning number of topics, computationally intensive

**Parameters**:
- `n_components`: Number of topics
- `max_iter`: Maximum iterations
- `learning_method`: 'online' (faster) or 'batch' (more accurate)

**When to Use**: Need interpretable topics, have large corpus, want probability distributions

---

### NMF (Non-Negative Matrix Factorization)

**Functionality**:
- Linear algebra factorization
- Decomposes document-term matrix into topic-term and document-topic matrices
- Enforces non-negativity constraint

**Evaluation**:
- ✅ **Strengths**: Faster than LDA, clearer topic separation, works well with short texts
- ✅ **Best for**: Quick topic discovery, short documents (tweets, reviews)
- ⚠️ **Limitations**: Less interpretable probabilistically, sensitive to initialization

**Parameters**:
- `n_components`: Number of topics
- `init`: Initialization method ('nndsvda' recommended)
- `max_iter`: Maximum iterations

**When to Use**: Need speed, short texts, clear topic separation

---

### BERTopic (Recommended for Future)

**Functionality**:
- Leverages BERT embeddings for semantic understanding
- Uses UMAP for dimensionality reduction
- HDBSCAN for clustering
- c-TF-IDF for topic representation

**Evaluation**:
- ✅ **Strengths**: State-of-the-art, semantic topics, dynamic topic modeling
- ✅ **Best for**: Semantic topic discovery, evolving topics over time
- ⚠️ **Limitations**: Requires more computational resources, needs sentence-transformers

**Installation**:
```bash
pip install bertopic
```

**Example Usage**:
```python
from bertopic import BERTopic

model = BERTopic(language="english", calculate_probabilities=True)
topics, probs = model.fit_transform(documents)
```

**When to Use**: Need semantic understanding, have computational resources, want best quality topics

---

### Coherence Metrics (Recommended Addition)

Measure topic quality:

```python
from gensim.models.coherencemodel import CoherenceModel

# Calculate coherence score
coherence_model = CoherenceModel(
    model=lda_model,
    texts=tokenized_docs,
    dictionary=dictionary,
    coherence='c_v'
)
coherence_score = coherence_model.get_coherence()
```

---

## 📊 Performance Metrics

### Classification Metrics

1. **Accuracy**: Overall correctness
   - Formula: `(TP + TN) / Total`
   - When to use: Balanced datasets

2. **Precision**: How many predicted positives are correct
   - Formula: `TP / (TP + FP)`
   - When to use: Cost of false positives is high

3. **Recall**: How many actual positives were found
   - Formula: `TP / (TP + FN)`
   - When to use: Cost of false negatives is high

4. **F1-Score**: Harmonic mean of precision and recall
   - Formula: `2 * (Precision * Recall) / (Precision + Recall)`
   - When to use: Need balance, imbalanced datasets

5. **ROC-AUC**: Area under ROC curve
   - Range: [0, 1], higher is better
   - When to use: Probability-based evaluation, threshold tuning

### Cross-Validation
- **Stratified K-Fold**: Maintains class distribution across folds
- **Benefits**: Reduces variance, detects overfitting
- **Recommendation**: Use 5-10 folds

---

## 🏗️ Project Structure

```
SentimentAnalysisTopicModellingPaper/
│
├── sentiment_analysis_app.py       # Main application file
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── ALGORITHMS.md                    # Detailed algorithm documentation
│
├── data/
│   └── text.csv                     # Sample dataset
│
├── models/                          # Saved models (created on first run)
│   ├── logistic_regression.pkl
│   └── ...
│
├── docs/                            # Additional documentation
│   ├── model_comparison.md
│   └── topic_modeling_guide.md
│
└── tests/                           # Unit tests
    └── test_preprocessing.py
```

---

## 🎨 Feature Extraction Comparison

### Current: TF-IDF
**Functionality**: Term Frequency - Inverse Document Frequency

**Evaluation**:
- ✅ **Strengths**: Simple, effective, interpretable
- ✅ **Best for**: Traditional ML models (SVM, Logistic Regression)
- ⚠️ **Limitations**: Doesn't capture word order or semantics

---

### Alternative: Count Vectorization
**Functionality**: Simple word counts

**Evaluation**:
- ✅ **Strengths**: Very simple, works well with Naive Bayes
- ⚠️ **Limitations**: No weighting for rare/common words

**When to Use**: Naive Bayes, LDA topic modeling

---

### Recommended: Word Embeddings

#### Word2Vec
```python
from gensim.models import Word2Vec

model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)
```

**Use Case**: Capture semantic similarity, clustering

#### GloVe
```python
# Use pre-trained GloVe embeddings
# Download from: https://nlp.stanford.edu/projects/glove/
```

**Use Case**: Transfer learning, semantic tasks

#### BERT Embeddings
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)
```

**Use Case**: State-of-the-art semantic representations, deep learning

---

## 🔧 Text Preprocessing Evaluation

### Current Pipeline
1. **Lowercasing**: ✅ Good
2. **URL Removal**: ✅ Good
3. **Digit Removal**: ⚠️ Consider keeping for some contexts
4. **Punctuation Removal**: ⚠️ Removes emojis (bad for sentiment)
5. **Stopword Removal**: ⚠️ Removes negations (e.g., "not good")
6. **Lemmatization**: ✅ Good (better than stemming)

### Recommendations

#### Keep Negations
```python
# Custom stopword list
stop_words = set(stopwords.words('english'))
negation_words = {'not', 'no', 'never', 'neither', 'nobody', 'nothing'}
stop_words = stop_words - negation_words
```

#### Handle Emojis
```python
import emoji

def extract_emojis(text):
    return [c for c in text if c in emoji.EMOJI_DATA]

# Or use emoji library for sentiment
from emoji import demojize
text = demojize(text)  # Convert emojis to :smile:, :angry:, etc.
```

#### Handle Contractions
```python
import contractions

def expand_contractions(text):
    return contractions.fix(text)  # don't -> do not
```

---

## 🚀 Future Enhancements

### High Priority
1. **Add XGBoost, LightGBM models**
2. **Implement BERTopic for topic modeling**
3. **Add confusion matrix visualization**
4. **Implement model persistence (save/load)**
5. **Add hyperparameter tuning (GridSearchCV)**

### Medium Priority
6. **BERT/RoBERTa fine-tuning**
7. **Add SHAP for model explainability**
8. **Implement active learning**
9. **Add multi-language support**
10. **Create REST API (FastAPI)**

### Low Priority
11. **Add LSTM/GRU models**
12. **Implement attention mechanisms**
13. **Add adversarial examples testing**
14. **Create Docker containerization**

---

## 📝 Example Workflows

### Workflow 1: Quick Sentiment Analysis
```python
# 1. Load data
data = pd.read_csv('text.csv')

# 2. Preprocess
data['text'] = data['text'].apply(clean_text).apply(preprocess_text)

# 3. Train baseline
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# 4. Evaluate
accuracy = model.score(X_test_tfidf, y_test)
```

### Workflow 2: Topic-Guided Sentiment Analysis
```python
# 1. Discover topics
topics = perform_topic_modeling(texts, method='lda', n_topics=5)

# 2. Assign topics to documents
doc_topics = get_document_topics(model, vectorizer, texts)

# 3. Train sentiment model per topic
for topic_id in range(5):
    topic_texts = get_texts_by_topic(texts, doc_topics, topic_id)
    # Train separate model for this topic
```

### Workflow 3: Production Deployment
```python
# 1. Train best model
best_model = GradientBoostingClassifier()
best_model.fit(X_train, y_train)

# 2. Save model
import joblib
joblib.dump(best_model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# 3. Load and predict
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
prediction = model.predict(vectorizer.transform([new_text]))
```

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

## 🙏 Acknowledgments

- NLTK for natural language processing tools
- scikit-learn for machine learning algorithms
- Streamlit for the interactive UI framework
- spaCy for advanced NLP capabilities
- TextBlob for sentiment analysis utilities

---

## 📚 References

### Sentiment Analysis
1. Pang, B., & Lee, L. (2008). Opinion mining and sentiment analysis. Foundations and Trends in Information Retrieval, 2(1–2), 1-135.
2. Liu, B. (2012). Sentiment analysis and opinion mining. Synthesis lectures on human language technologies, 5(1), 1-167.

### Topic Modeling
3. Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet allocation. Journal of machine Learning research, 3(Jan), 993-1022.
4. Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization. Nature, 401(6755), 788-791.

### Deep Learning
5. Devlin, J., et al. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
6. Liu, Y., et al. (2019). Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.
