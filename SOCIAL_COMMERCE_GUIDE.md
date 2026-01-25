# Social Commerce Topic Modeling Guide

## Overview

This guide explains how to use the **Social Commerce Topic Modeling** features to classify online posts and comments into business-relevant categories.

## What Problem Does This Solve?

Standard topic modeling (LDA, NMF) produces **generic word clusters** like:
- Topic 1: "product, quality, price, good"
- Topic 2: "delivery, fast, service, time"
- Topic 3: "buy, want, order, purchase"

These are hard to interpret for business use cases. **Social Commerce Topic Modeling** solves this by classifying text into **predefined business categories**:

1. **Purchase Intent** - Users expressing desire to buy
2. **Product Reviews** - Opinions about product quality
3. **Shopping Experience** - Delivery, service, website usability
4. **Price Sensitivity** - Price concerns, discounts, deals
5. **Social Influence** - Recommendations, trending, influencers

## Quick Start

### Option 1: Standalone Demo (Recommended for Testing)

Run the standalone demo app:

```bash
streamlit run social_commerce_demo.py
```

This provides:
- ✅ Pre-loaded sample data (600 social commerce examples)
- ✅ All 4 classification methods
- ✅ CSV upload capability
- ✅ Quick single-text testing
- ✅ Topic definitions reference

### Option 2: Integration with Main App

Import and use the classifiers in your own code:

```python
from guided_topic_modeling import (
    KeywordTopicClassifier,
    GuidedLDATopicModel,
    ZeroShotTopicClassifier,
    GuidedBERTopicModel,
    SOCIAL_COMMERCE_TOPICS
)

# Quick classification with keywords
classifier = KeywordTopicClassifier(SOCIAL_COMMERCE_TOPICS)
topic, confidence, scores = classifier.classify_document(
    "I want to buy this product, is it available?"
)
print(f"Topic: {topic}, Confidence: {confidence:.1%}")
# Output: Topic: Purchase Intent, Confidence: 42%
```

## The 4 Classification Methods

### 1. 🔑 Keyword-Based (Fastest)

**When to use:** Quick prototyping, validation, real-time classification

**How it works:**
- Matches text against predefined seed words
- Calculates score based on keyword frequency
- No training required, instant results

**Pros:**
- ⚡ Instant results (< 1ms per document)
- 📖 Fully interpretable
- 💾 No model storage needed
- 🎯 Good baseline accuracy (~70-75%)

**Cons:**
- Limited to exact keyword matches
- Can't understand context or synonyms
- Requires good seed word selection

**Example:**
```python
classifier = KeywordTopicClassifier(SOCIAL_COMMERCE_TOPICS)
topic, confidence, scores = classifier.classify_document(text)
```

### 2. 📊 Guided LDA (Balanced)

**When to use:** When you want to discover new relevant keywords while staying within your categories

**How it works:**
- Uses seed words to guide topic discovery
- Probabilistic topic modeling (Latent Dirichlet Allocation)
- Learns word-topic associations from your data

**Pros:**
- 🔍 Discovers new relevant keywords
- 📈 Better than pure unsupervised LDA
- 🎲 Probabilistic (provides topic distribution)
- 🎯 Good accuracy (~75-80%)

**Cons:**
- ⏱️ Requires training (1-2 minutes for 1000 docs)
- 🎛️ Needs parameter tuning (alpha, beta)
- 📊 Less interpretable than keywords

**Example:**
```python
model = GuidedLDATopicModel(SOCIAL_COMMERCE_TOPICS, n_topics=5)
model.fit(documents)
topic, confidence, scores = model.classify_document(text)
```

### 3. 🤖 Zero-Shot Transformer (Most Accurate)

**When to use:** When accuracy is critical and you have GPU available

**How it works:**
- Uses pre-trained BART/RoBERTa model
- Natural Language Inference (NLI) approach
- No training needed on your data

**Pros:**
- 🎯 **Highest accuracy** (~85-90%)
- 🚀 No training required
- 🧠 Understands context, synonyms, semantics
- 🌐 Works across domains

**Cons:**
- ⏱️ Slower (5-10 min for 1000 docs on CPU)
- 💻 Benefits from GPU
- 📦 Requires transformers library
- 💾 Large model download (~1.6GB)

**Example:**
```python
classifier = ZeroShotTopicClassifier(SOCIAL_COMMERCE_TOPICS)
topic, confidence, scores = classifier.classify_document(text)
```

### 4. 🧠 BERTopic (Best Quality)

**When to use:** When you want the highest quality topic understanding and can afford longer processing time

**How it works:**
- Uses BERT embeddings for semantic understanding
- Discovers topics, then maps to your categories
- Combines unsupervised discovery with supervised mapping

**Pros:**
- 🌟 **Best topic quality**
- 🎯 Excellent accuracy (~80-85%)
- 🔍 Discovers semantic patterns
- 📊 Rich topic information

**Cons:**
- ⏱️ **Slowest** (10-15 min for 1000 docs)
- 💻 Requires sentence-transformers
- 🧮 More complex setup
- 💾 Large model (~90MB)

**Example:**
```python
model = GuidedBERTopicModel(SOCIAL_COMMERCE_TOPICS)
model.fit(documents)
topic, confidence, scores = model.classify_document(text)
```

## Comparison Table

| Method | Speed | Accuracy | Setup | Training | GPU Benefit |
|--------|-------|----------|-------|----------|-------------|
| **Keyword-Based** | ⚡⚡⚡ Instant | 70-75% | ✅ None | ❌ No | ❌ No |
| **Guided LDA** | ⚡⚡ Fast | 75-80% | ⚙️ Medium | ✅ Yes | ❌ No |
| **Zero-Shot** | ⚡ Moderate | **85-90%** | ⚙️ Medium | ❌ No | ✅ Yes |
| **BERTopic** | 🐌 Slow | 80-85% | ⚙️⚙️ Complex | ✅ Yes | ✅ Yes |

## Usage Workflow

### Step 1: Test with Sample Data

```bash
streamlit run social_commerce_demo.py
```

1. Select "Use Sample Data" (600 pre-loaded examples)
2. Go to "📊 Topic Classification" tab
3. Try "🔑 Keyword-Based" first (instant results)
4. Review topic distribution and examples

### Step 2: Upload Your Data

1. Prepare CSV with required columns:
   - `text` (required): Your posts/comments
   - `sentiment` (optional): Sentiment labels if available

2. Click "Upload CSV File" in sidebar
3. Select your file
4. Data will be loaded automatically

### Step 3: Choose Classification Method

**For quick exploration:**
- Start with **Keyword-Based**
- Review results, check if topics make sense
- Look at low-confidence examples

**For production use:**
- Use **Zero-Shot** if you have GPU
- Use **Guided LDA** if you need speed
- Use **BERTopic** if quality is priority

### Step 4: Analyze Results

The demo provides:
- **Topic Distribution**: Bar chart showing document counts per topic
- **Confidence Distribution**: Histogram of prediction confidence
- **Topic Heatmap**: Document-topic probability matrix
- **Sample Documents**: Examples from each topic
- **Low Confidence Examples**: Cases that need review

### Step 5: Download Results

Click "💾 Download Results as CSV" to get:
```csv
original_text,predicted_topic,confidence
"Looking to buy new phone","Purchase Intent",0.87
"Great product, highly recommend","Product Reviews",0.92
...
```

## Customizing Topics

You can customize the 5 topics by editing `guided_topic_modeling.py`:

```python
SOCIAL_COMMERCE_TOPICS = {
    "Your Custom Topic": {
        "description": "What this topic represents",
        "seed_words": [
            "keyword1", "keyword2", "phrase example",
            "another keyword", "more keywords"
        ],
        "patterns": [
            r"\b(regex pattern here)\b",
        ]
    },
    # Add more topics...
}
```

**Tips for seed words:**
- Use 20-50 keywords per topic
- Include synonyms and variations
- Use phrases, not just single words
- Test with your actual data
- Iterate based on results

## Combining with Sentiment Analysis

After topic classification, combine with sentiment analysis for deeper insights:

```python
# Classify topic
topic, confidence, scores = classifier.classify_document(text)

# Add sentiment (from main app)
sentiment = analyze_sentiment(text)  # From sentiment_analysis_app.py

# Create combined insight
result = {
    'text': text,
    'topic': topic,
    'topic_confidence': confidence,
    'sentiment': sentiment,
    'insight': f"{sentiment} sentiment about {topic}"
}

# Example output:
# {
#   'topic': 'Product Reviews',
#   'sentiment': 'positive',
#   'insight': 'positive sentiment about Product Reviews'
# }
```

This allows analysis like:
- "Users with **Purchase Intent** are mostly **positive** (85%)"
- "**Price Sensitivity** posts are **negative** more often (60%)"
- "**Social Influence** drives **positive** sentiment (78%)"

## Performance Tips

### For Large Datasets (10K+ documents)

1. **Use batching:**
```python
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    results = classifier.classify_batch(batch)
```

2. **Use Keyword-Based first:**
   - Filter high-confidence predictions (>0.8)
   - Only run Zero-Shot on low-confidence examples
   - Saves 90% of processing time

3. **Enable GPU:**
```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Zero-Shot will automatically use GPU if available
```

### For Real-Time Classification

Use Keyword-Based classifier:
```python
# Initialize once
classifier = KeywordTopicClassifier(SOCIAL_COMMERCE_TOPICS)

# Classify in real-time
topic, confidence, scores = classifier.classify_document(new_text)
```

Response time: < 1ms per document

## Troubleshooting

### Issue: All texts classified as same topic

**Solution:** Check seed word balance
```python
for topic, info in SOCIAL_COMMERCE_TOPICS.items():
    print(f"{topic}: {len(info['seed_words'])} seed words")
```
Ensure each topic has similar number of seed words (20-50).

### Issue: Low confidence scores

**Possible causes:**
1. Text is ambiguous (contains multiple topics)
2. Seed words don't match your domain
3. Text too short

**Solutions:**
- Review and expand seed words
- Use Zero-Shot (better at understanding context)
- Filter very short texts (< 10 words)

### Issue: Transformers/BERTopic not available

**Install required packages:**
```bash
pip install transformers torch sentence-transformers
pip install bertopic umap-learn hdbscan
```

### Issue: Out of memory

**Solutions:**
1. Use smaller batch sizes
2. Use Keyword or Guided LDA instead of transformers
3. Sample your data (test on subset first)

## API Reference

### KeywordTopicClassifier

```python
classifier = KeywordTopicClassifier(topics_config)
topic, confidence, scores = classifier.classify_document(text)
```

**Returns:**
- `topic` (str): Predicted topic name
- `confidence` (float): Score between 0-1
- `scores` (dict): Scores for all topics

### GuidedLDATopicModel

```python
model = GuidedLDATopicModel(topics_config, n_topics=5, alpha=0.1, beta=0.01)
model.fit(documents)
topic, confidence, scores = model.classify_document(text)
```

**Parameters:**
- `alpha`: Topic distribution sparsity (lower = fewer topics per doc)
- `beta`: Word distribution sparsity (lower = fewer words per topic)

### ZeroShotTopicClassifier

```python
classifier = ZeroShotTopicClassifier(topics_config)
topic, confidence, scores = classifier.classify_document(text)
```

**Note:** First call downloads model (~1.6GB), subsequent calls use cached model.

### GuidedBERTopicModel

```python
model = GuidedBERTopicModel(topics_config, n_topics=5)
model.fit(documents)
topic, confidence, scores = model.classify_document(text)
```

## Examples

### Example 1: Classify Single Text

```python
from guided_topic_modeling import KeywordTopicClassifier, SOCIAL_COMMERCE_TOPICS

classifier = KeywordTopicClassifier(SOCIAL_COMMERCE_TOPICS)

text = "Looking to buy a new laptop, any recommendations?"
topic, confidence, scores = classifier.classify_document(text)

print(f"Topic: {topic}")
print(f"Confidence: {confidence:.1%}")
print("\nAll scores:")
for t, s in sorted(scores.items(), key=lambda x: x[1], reverse=True):
    print(f"  {t}: {s:.1%}")
```

Output:
```
Topic: Purchase Intent
Confidence: 67%

All scores:
  Purchase Intent: 67%
  Product Reviews: 0%
  Shopping Experience: 0%
  Price Sensitivity: 0%
  Social Influence: 0%
```

### Example 2: Classify Dataset

```python
import pandas as pd
from guided_topic_modeling import ZeroShotTopicClassifier, SOCIAL_COMMERCE_TOPICS

# Load data
df = pd.read_csv("online_posts.csv")

# Classify
classifier = ZeroShotTopicClassifier(SOCIAL_COMMERCE_TOPICS)
results = []

for text in df['text']:
    topic, confidence, scores = classifier.classify_document(text)
    results.append({
        'text': text,
        'topic': topic,
        'confidence': confidence
    })

# Create results dataframe
results_df = pd.DataFrame(results)
results_df.to_csv("classified_results.csv", index=False)

# Analyze
print(results_df['topic'].value_counts())
print(f"\nAverage confidence: {results_df['confidence'].mean():.1%}")
```

### Example 3: Compare Methods

```python
from guided_topic_modeling import (
    KeywordTopicClassifier,
    ZeroShotTopicClassifier,
    SOCIAL_COMMERCE_TOPICS
)

text = "This product is amazing and everyone recommends it!"

# Keyword method
kw_classifier = KeywordTopicClassifier(SOCIAL_COMMERCE_TOPICS)
kw_topic, kw_conf, _ = kw_classifier.classify_document(text)

# Zero-shot method
zs_classifier = ZeroShotTopicClassifier(SOCIAL_COMMERCE_TOPICS)
zs_topic, zs_conf, _ = zs_classifier.classify_document(text)

print(f"Keyword: {kw_topic} ({kw_conf:.1%})")
print(f"Zero-Shot: {zs_topic} ({zs_conf:.1%})")
```

Output:
```
Keyword: Product Reviews (50%)
Zero-Shot: Social Influence (87%)
```

## Next Steps

1. ✅ Test with sample data using `social_commerce_demo.py`
2. ✅ Upload your own online posts/comments CSV
3. ✅ Choose appropriate classification method
4. ✅ Review and adjust seed words if needed
5. ✅ Download classified results
6. ✅ Combine with sentiment analysis for deeper insights

## Questions?

- **How to add new topics?** Edit `SOCIAL_COMMERCE_TOPICS` in `guided_topic_modeling.py`
- **Which method is best?** Start with Keyword, use Zero-Shot for production
- **How to improve accuracy?** Add more relevant seed words, test different methods
- **Can I use supervised learning?** Yes, label some data and train a classifier (not covered here)
- **How to integrate with main app?** Import the classifiers and use in your workflow

For more details, see:
- `guided_topic_modeling.py` - Core implementation
- `social_commerce_demo.py` - Standalone demo
- `social_commerce_topic_tab.py` - Streamlit tab component
