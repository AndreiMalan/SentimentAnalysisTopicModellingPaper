"""
Sentiment Analysis Module
=========================
Emotion-based sentiment analysis using:
1. Pre-trained BERT model (distilbert-base-uncased-emotion) - 6 emotions
2. VADER sentiment intensity
3. TextBlob polarity and subjectivity

The 6 emotions: sadness, joy, love, anger, fear, surprise.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

from config.constructs import EMOTION_LABELS

# Optional transformer imports
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ========================================================================
# BERT Emotion Model
# ========================================================================

_BERT_MODEL = None
_BERT_TOKENIZER = None
_DEVICE = None


def _load_bert_emotion():
    global _BERT_MODEL, _BERT_TOKENIZER, _DEVICE
    if _BERT_MODEL is not None:
        return _BERT_MODEL, _BERT_TOKENIZER, _DEVICE
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("pip install transformers torch")
    model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
    _BERT_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    _BERT_MODEL = AutoModelForSequenceClassification.from_pretrained(model_name)
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _BERT_MODEL.to(_DEVICE)
    _BERT_MODEL.eval()
    return _BERT_MODEL, _BERT_TOKENIZER, _DEVICE


def predict_emotions_bert(
    texts: List[str], batch_size: int = 32, progress_callback=None,
) -> pd.DataFrame:
    """Predict 6 emotions for a list of texts using BERT.

    Returns DataFrame with columns:
        emotion, emotion_id, prob_sadness ... prob_surprise
    """
    model, tokenizer, device = _load_bert_emotion()
    all_preds = []
    all_probs = []
    total = len(texts)

    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
        all_preds.extend(torch.argmax(probs, dim=-1).cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        if progress_callback:
            progress_callback(min(1.0, (start + batch_size) / total))

    preds = np.array(all_preds)
    probs = np.array(all_probs)

    df = pd.DataFrame({
        "emotion_id": preds,
        "emotion": [EMOTION_LABELS[p] for p in preds],
    })
    for eid, ename in EMOTION_LABELS.items():
        df[f"prob_{ename}"] = probs[:, eid]

    return df


# ========================================================================
# VADER
# ========================================================================

_VADER = None


def _get_vader():
    global _VADER
    if _VADER is None:
        _VADER = SentimentIntensityAnalyzer()
    return _VADER


def vader_sentiment(texts: List[str]) -> pd.DataFrame:
    """Compute VADER compound, positive, neutral, negative scores."""
    sia = _get_vader()
    rows = []
    for text in texts:
        scores = sia.polarity_scores(text)
        label = "positive" if scores["compound"] > 0.05 else (
            "negative" if scores["compound"] < -0.05 else "neutral"
        )
        rows.append({
            "vader_compound": scores["compound"],
            "vader_pos": scores["pos"],
            "vader_neu": scores["neu"],
            "vader_neg": scores["neg"],
            "vader_label": label,
        })
    return pd.DataFrame(rows)


# ========================================================================
# TextBlob
# ========================================================================

def textblob_sentiment(texts: List[str]) -> pd.DataFrame:
    """Compute TextBlob polarity and subjectivity."""
    rows = []
    for text in texts:
        blob = TextBlob(text)
        rows.append({
            "tb_polarity": blob.sentiment.polarity,
            "tb_subjectivity": blob.sentiment.subjectivity,
            "tb_label": (
                "positive" if blob.sentiment.polarity > 0.05 else
                "negative" if blob.sentiment.polarity < -0.05 else "neutral"
            ),
        })
    return pd.DataFrame(rows)


# ========================================================================
# Combined sentiment pipeline
# ========================================================================

def run_sentiment_pipeline(
    texts: List[str],
    use_bert: bool = True,
    progress_callback=None,
) -> pd.DataFrame:
    """Run all sentiment methods and merge results.

    Returns DataFrame with columns from BERT, VADER, and TextBlob.
    """
    dfs = []

    # BERT emotions
    if use_bert and TRANSFORMERS_AVAILABLE:
        if progress_callback:
            progress_callback("BERT emotions", 0.0)
        bert_df = predict_emotions_bert(
            texts,
            progress_callback=lambda p: progress_callback("BERT emotions", p) if progress_callback else None,
        )
        dfs.append(bert_df)
    else:
        # fallback: create empty columns
        dfs.append(pd.DataFrame({
            "emotion_id": [None] * len(texts),
            "emotion": ["unavailable"] * len(texts),
        }))

    # VADER
    if progress_callback:
        progress_callback("VADER", 0.5)
    dfs.append(vader_sentiment(texts))

    # TextBlob
    if progress_callback:
        progress_callback("TextBlob", 0.8)
    dfs.append(textblob_sentiment(texts))

    combined = pd.concat(dfs, axis=1)
    if progress_callback:
        progress_callback("Done", 1.0)
    return combined
