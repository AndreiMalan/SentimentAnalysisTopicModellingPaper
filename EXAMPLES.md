# Usage Examples

This document provides practical examples for using the Sentiment Analysis & Topic Modeling application.

## Table of Contents
- [Basic Usage](#basic-usage)
- [Sentiment Analysis Examples](#sentiment-analysis-examples)
- [Topic Modeling Examples](#topic-modeling-examples)
- [Advanced Usage](#advanced-usage)
- [Custom Model Training](#custom-model-training)
- [Production Deployment](#production-deployment)

---

## Basic Usage

### Running the Streamlit App

```bash
# Navigate to project directory
cd SentimentAnalysisTopicModellingPaper

# Run the app
streamlit run sentiment_analysis_app.py
```

The app will open in your browser at `http://localhost:8501`

---

## Sentiment Analysis Examples

### Example 1: Basic Sentiment Prediction

```python
from sentiment_analysis_app import clean_text, preprocess_text, get_sentiment_models
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Sample text
text = "I absolutely love this product! It's amazing and works perfectly!"

# Preprocess
cleaned = clean_text(text)
processed = preprocess_text(cleaned)

# Create vectorizer and transform
vectorizer = TfidfVectorizer(max_features=1000)

# Note: In practice, you'd fit on training data
# This is just for demonstration
vectorizer.fit([processed])
text_vector = vectorizer.transform([processed])

# Load model and predict
models = get_sentiment_models()
lr_model = models['Logistic Regression']

# Train on your data first
# lr_model.fit(X_train, y_train)

# Predict
# prediction = lr_model.predict(text_vector)
# probabilities = lr_model.predict_proba(text_vector)
```

### Example 2: Batch Sentiment Analysis

```python
import pandas as pd
from sentiment_analysis_app import clean_text, preprocess_text

# Load your data
data = pd.read_csv('customer_reviews.csv')

# Preprocess all texts
data['cleaned_text'] = data['review'].apply(clean_text)
data['processed_text'] = data['cleaned_text'].apply(preprocess_text)

# Vectorize
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))

X = vectorizer.fit_transform(data['processed_text'])

# Train model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X, data['sentiment'])

# Predict on new data
new_reviews = ["This is terrible", "Best purchase ever!", "Okay product"]
new_cleaned = [clean_text(r) for r in new_reviews]
new_processed = [preprocess_text(r) for r in new_cleaned]
new_vectors = vectorizer.transform(new_processed)

predictions = model.predict(new_vectors)
probabilities = model.predict_proba(new_vectors)

print("Predictions:", predictions)
print("Probabilities:", probabilities)
```

### Example 3: Model Comparison

```python
from sentiment_analysis_app import get_sentiment_models, evaluate_model_optimized
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Load and prepare data
data = pd.read_csv('text.csv')
# ... preprocess data ...

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['processed_text'],
    data['sentiment'],
    test_size=0.2,
    random_state=42
)

# Vectorize
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Get all models
models = get_sentiment_models()
all_classes = np.arange(len(np.unique(y_train)))

# Compare models
results = {}
for name, model in models.items():
    print(f"Training {name}...")
    accuracy, precision, recall, f1, roc_auc = evaluate_model_optimized(
        model, X_train_vec, X_test_vec, y_train, y_test, all_classes, calc_auc=True
    )

    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

# Display results
results_df = pd.DataFrame(results).T
print(results_df.sort_values('f1', ascending=False))
```

### Example 4: Sentiment Intensity & Subjectivity

```python
from sentiment_analysis_app import get_sentiment_intensity, get_subjectivity

texts = [
    "This is the best day ever!",
    "The weather is nice today.",
    "I absolutely hate this terrible product!",
    "The cat sat on the mat."
]

for text in texts:
    intensity = get_sentiment_intensity(text)
    subjectivity = get_subjectivity(text)

    print(f"Text: {text}")
    print(f"  Intensity: {intensity:.3f}")
    print(f"  Subjectivity: {subjectivity:.3f}")
    print()
```

Output:
```
Text: This is the best day ever!
  Intensity: 0.843
  Subjectivity: 1.000

Text: The weather is nice today.
  Intensity: 0.431
  Subjectivity: 0.600

Text: I absolutely hate this terrible product!
  Intensity: -0.839
  Subjectivity: 1.000

Text: The cat sat on the mat.
  Intensity: 0.000
  Subjectivity: 0.000
```

### Example 5: Aspect-Based Sentiment Analysis

```python
from sentiment_analysis_app import aspect_sentiment_analysis

review = """
The food at this restaurant was delicious, but the service was terrible.
The ambiance is nice and cozy. However, the prices are too high.
"""

aspects = aspect_sentiment_analysis(review)

print("Aspect-Based Sentiment Analysis:")
for aspect, sentiment in aspects.items():
    sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
    print(f"  {aspect}: {sentiment:.3f} ({sentiment_label})")
```

Output:
```
Aspect-Based Sentiment Analysis:
  The food: 0.850 (Positive)
  this restaurant: 0.000 (Neutral)
  the service: -0.650 (Negative)
  The ambiance: 0.500 (Positive)
  the prices: -0.400 (Negative)
```

---

## Topic Modeling Examples

### Example 6: LDA Topic Modeling

```python
from sentiment_analysis_app import perform_topic_modeling, get_document_topics
import pandas as pd

# Load documents
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with many layers",
    "Python is a popular programming language for data science",
    "Natural language processing helps computers understand human language",
    "Computer vision enables machines to interpret visual information",
    # ... more documents ...
]

# Perform LDA topic modeling
topic_model, vectorizer, feature_names, topics_dict = perform_topic_modeling(
    documents,
    n_topics=3,
    method='lda',
    n_top_words=10
)

# Display topics
print("Discovered Topics (LDA):")
for topic_name, words in topics_dict.items():
    print(f"\n{topic_name}:")
    print(f"  Keywords: {', '.join(words)}")

# Get topic distribution for a new document
new_doc = ["Python is used for machine learning and AI applications"]
doc_topics = get_document_topics(topic_model, vectorizer, new_doc, top_n=3)

print("\nTopic Distribution for New Document:")
for topic_idx, prob in doc_topics[0]:
    print(f"  Topic {topic_idx + 1}: {prob:.3f}")
```

### Example 7: NMF Topic Modeling

```python
from sentiment_analysis_app import perform_topic_modeling

# Short texts (e.g., tweets)
tweets = [
    "Just bought the new iPhone! Love it! #apple #iphone",
    "Android phones are better value for money #android",
    "Breaking: Stock market hits all-time high #stocks #finance",
    "Investing in cryptocurrency is risky but potentially profitable #crypto",
    "Just watched an amazing movie! Highly recommend! #movies",
    # ... more tweets ...
]

# NMF works better with short texts
topic_model, vectorizer, feature_names, topics_dict = perform_topic_modeling(
    tweets,
    n_topics=3,
    method='nmf',
    n_top_words=8
)

# Display topics
print("Discovered Topics (NMF):")
for topic_name, words in topics_dict.items():
    print(f"\n{topic_name}:")
    print(f"  Keywords: {', '.join(words)}")
```

### Example 8: Topic-Guided Sentiment Analysis

```python
from sentiment_analysis_app import perform_topic_modeling, get_document_topics
import numpy as np

# Combine topic modeling with sentiment analysis
documents = ["..."]  # Your documents
sentiments = [...]  # Corresponding sentiment labels

# Get topics
topic_model, vectorizer, _, topics_dict = perform_topic_modeling(
    documents,
    n_topics=5,
    method='lda'
)

# Get dominant topic for each document
doc_topics = get_document_topics(topic_model, vectorizer, documents, top_n=1)
dominant_topics = [topics[0][0] for topics in doc_topics]

# Analyze sentiment by topic
import pandas as pd
df = pd.DataFrame({
    'document': documents,
    'sentiment': sentiments,
    'topic': dominant_topics
})

# Average sentiment per topic
sentiment_by_topic = df.groupby('topic')['sentiment'].mean()

print("Average Sentiment by Topic:")
for topic_id, avg_sentiment in sentiment_by_topic.items():
    print(f"  Topic {topic_id + 1}: {avg_sentiment:.3f}")
    print(f"    Keywords: {', '.join(topics_dict[f'Topic {topic_id + 1}'])}")
```

---

## Advanced Usage

### Example 9: Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np

# Define parameter grid
param_grid = {
    'C': [0.1, 1.0, 10.0, 100.0],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [200, 500]
}

# Create model
lr = LogisticRegression(random_state=42, class_weight='balanced')

# Grid search
grid_search = GridSearchCV(
    lr,
    param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

# Fit
grid_search.fit(X_train_vec, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)
print("Best F1 Score:", grid_search.best_score_)

# Use best model
best_model = grid_search.best_estimator_
```

### Example 10: Cross-Validation with Multiple Metrics

```python
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

# Define multiple metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision_weighted',
    'recall': 'recall_weighted',
    'f1': 'f1_weighted'
}

# Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Cross-validation
cv_results = cross_validate(
    rf,
    X_train_vec,
    y_train,
    cv=5,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=True
)

# Display results
import pandas as pd
results_df = pd.DataFrame(cv_results)
print("Cross-Validation Results:")
print(results_df.mean())
```

### Example 11: Feature Importance Analysis

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_vec, y_train)

# Get feature importance
feature_importance = rf.feature_importances_
feature_names = vectorizer.get_feature_names_out()

# Sort by importance
indices = np.argsort(feature_importance)[-20:]  # Top 20

# Plot
plt.figure(figsize=(10, 8))
plt.barh(range(len(indices)), feature_importance[indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Feature Importance')
plt.title('Top 20 Most Important Features')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance plot saved!")
```

### Example 12: Ensemble Model

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

# Create base models
lr = LogisticRegression(max_iter=200, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
nb = MultinomialNB()

# Create ensemble
ensemble = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('rf', rf),
        ('nb', nb)
    ],
    voting='soft',  # Use probability voting
    weights=[1, 2, 1]  # Give more weight to Random Forest
)

# Train
ensemble.fit(X_train_vec, y_train)

# Evaluate
from sklearn.metrics import classification_report
y_pred = ensemble.predict(X_test_vec)
print(classification_report(y_test, y_pred))
```

---

## Custom Model Training

### Example 13: Training Custom XGBoost Model

```python
# First install: pip install xgboost

import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score

# Create DMatrix (XGBoost's internal data structure)
dtrain = xgb.DMatrix(X_train_vec, label=y_train)
dtest = xgb.DMatrix(X_test_vec, label=y_test)

# Parameters
params = {
    'objective': 'multi:softprob',
    'num_class': 6,  # Number of sentiment classes
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'mlogloss',
    'seed': 42
}

# Train
num_rounds = 200
evals = [(dtrain, 'train'), (dtest, 'test')]
model = xgb.train(
    params,
    dtrain,
    num_rounds,
    evals=evals,
    early_stopping_rounds=10,
    verbose_eval=20
)

# Predict
y_pred_proba = model.predict(dtest)
y_pred = y_pred_proba.argmax(axis=1)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

# Feature importance
xgb.plot_importance(model, max_num_features=20)
plt.tight_layout()
plt.savefig('xgboost_importance.png')
```

### Example 14: Training with LightGBM

```python
# First install: pip install lightgbm

import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score

# Create Dataset
train_data = lgb.Dataset(X_train_vec, label=y_train)
test_data = lgb.Dataset(X_test_vec, label=y_test, reference=train_data)

# Parameters
params = {
    'objective': 'multiclass',
    'num_class': 6,
    'metric': 'multi_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'seed': 42
}

# Train
model = lgb.train(
    params,
    train_data,
    num_boost_round=200,
    valid_sets=[train_data, test_data],
    valid_names=['train', 'test'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=10),
        lgb.log_evaluation(period=20)
    ]
)

# Predict
y_pred_proba = model.predict(X_test_vec)
y_pred = y_pred_proba.argmax(axis=1)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
```

---

## Production Deployment

### Example 15: Save and Load Models

```python
import joblib

# Train model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# Save model and vectorizer
joblib.dump(model, 'models/sentiment_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

print("Model saved successfully!")

# Load model
loaded_model = joblib.load('models/sentiment_model.pkl')
loaded_vectorizer = joblib.load('models/vectorizer.pkl')

# Predict with loaded model
new_text = ["This product is amazing!"]
new_vec = loaded_vectorizer.transform(new_text)
prediction = loaded_model.predict(new_vec)

print(f"Prediction: {prediction}")
```

### Example 16: FastAPI REST API

```python
# Create file: api.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from sentiment_analysis_app import clean_text, preprocess_text

# Load model
model = joblib.load('models/sentiment_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

app = FastAPI()

class TextInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    sentiment: int
    sentiment_label: str
    probabilities: dict

EMOTION_MAPPING = {
    0: "sadness", 1: "joy", 2: "love",
    3: "anger", 4: "fear", 5: "surprise"
}

@app.post("/predict", response_model=PredictionOutput)
def predict_sentiment(input_data: TextInput):
    # Preprocess
    cleaned = clean_text(input_data.text)
    processed = preprocess_text(cleaned)

    # Vectorize
    text_vec = vectorizer.transform([processed])

    # Predict
    prediction = model.predict(text_vec)[0]
    probabilities = model.predict_proba(text_vec)[0]

    # Format probabilities
    prob_dict = {
        EMOTION_MAPPING[i]: float(prob)
        for i, prob in enumerate(probabilities)
    }

    return PredictionOutput(
        sentiment=int(prediction),
        sentiment_label=EMOTION_MAPPING[prediction],
        probabilities=prob_dict
    )

# Run with: uvicorn api:app --reload
```

### Example 17: Batch Processing Script

```python
# batch_process.py

import pandas as pd
import joblib
from sentiment_analysis_app import clean_text, preprocess_text
from tqdm import tqdm

# Load model
model = joblib.load('models/sentiment_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# Load data
input_file = 'data/to_process.csv'
output_file = 'data/processed_results.csv'

df = pd.read_csv(input_file)

# Process in batches
batch_size = 1000
predictions = []
probabilities = []

for i in tqdm(range(0, len(df), batch_size)):
    batch = df.iloc[i:i+batch_size]

    # Preprocess
    cleaned = batch['text'].apply(clean_text)
    processed = cleaned.apply(preprocess_text)

    # Vectorize
    batch_vec = vectorizer.transform(processed)

    # Predict
    batch_pred = model.predict(batch_vec)
    batch_proba = model.predict_proba(batch_vec)

    predictions.extend(batch_pred)
    probabilities.extend(batch_proba.tolist())

# Add results
df['sentiment_prediction'] = predictions
df['probabilities'] = probabilities

# Save
df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")
```

---

## Tips and Best Practices

### 1. Data Preprocessing
- Always clean text before processing
- Handle missing values appropriately
- Consider domain-specific preprocessing (e.g., keep hashtags for social media)

### 2. Model Selection
- Start with simple models (Logistic Regression, Naive Bayes)
- Use cross-validation to evaluate performance
- Try ensemble methods for better accuracy

### 3. Hyperparameter Tuning
- Use GridSearchCV or RandomizedSearchCV
- Focus on most important parameters first
- Monitor for overfitting

### 4. Production Deployment
- Save models and vectorizers together
- Version your models
- Monitor performance in production
- Implement logging and error handling

### 5. Performance Optimization
- Use batch processing for large datasets
- Consider using GPU for deep learning models
- Cache frequently used data

---

For more examples and documentation, see:
- README.md
- ALGORITHMS.md
- Official documentation: [link to docs]
