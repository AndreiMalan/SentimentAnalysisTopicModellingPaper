"""
Data Cleaning Pipeline
======================
Handles the full text-cleaning workflow for YouTube comments:
1. Fix encoding / mojibake (UTF-8 emoji artifacts)
2. Remove emoji characters
3. Detect language and keep only English comments
4. Standard NLP preprocessing (lowercase, URLs, punctuation, stopwords, lemmatization)
"""

import re
import string
import pandas as pd
import numpy as np

# --- optional heavy imports with graceful fallback ---
try:
    import ftfy
    FTFY_AVAILABLE = True
except ImportError:
    FTFY_AVAILABLE = False

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 42  # reproducibility
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    import emoji
    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False

try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_str(text) -> str:
    """Safely convert any value to a usable string.

    Returns "" for NaN, None, the literal string 'nan', or empty values.
    """
    if text is None:
        return ""
    if isinstance(text, float) and np.isnan(text):
        return ""
    s = str(text).strip()
    if s.lower() == "nan" or s.lower() == "none" or s == "":
        # Only treat as empty if it was a type-conversion artefact.
        # A real comment typed "nan" by a human is extremely unlikely.
        if not isinstance(text, str):
            return ""
        # If the original was already a string, check if it's literally "nan"
        if text.strip().lower() in ("nan", "none", ""):
            return ""
    return str(text)


# ---------------------------------------------------------------------------
# Step 1: Fix encoding / mojibake
# ---------------------------------------------------------------------------

def fix_encoding(text) -> str:
    """Fix mojibake caused by YouTube API encoding issues."""
    text = _to_str(text)
    if not text:
        return ""
    if FTFY_AVAILABLE:
        return ftfy.fix_text(text)
    return text


# ---------------------------------------------------------------------------
# Step 2: Remove emojis
# ---------------------------------------------------------------------------

def remove_emojis(text) -> str:
    """Remove emoji characters from text."""
    text = _to_str(text)
    if not text:
        return ""
    if EMOJI_AVAILABLE:
        text = emoji.replace_emoji(text, replace="")
    else:
        _fallback = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U0001F900-\U0001F9FF"
            "\U0001FA00-\U0001FA6F"
            "\U0001FA70-\U0001FAFF"
            "\u200d"
            "\ufe0f"
            "]+",
            flags=re.UNICODE,
        )
        text = _fallback.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Step 3: Language detection
# ---------------------------------------------------------------------------

def detect_language(text) -> str:
    """Detect language of a text string. Returns ISO 639-1 code or 'unknown'."""
    text = _to_str(text)
    if not text:
        return "unknown"
    if not LANGDETECT_AVAILABLE:
        return "unknown"
    # Strip non-letter chars for cleaner detection
    clean = re.sub(r"[^a-zA-Z\s]", "", text)
    if len(clean.split()) < 3:
        # Too short to detect reliably — check if it's at least Latin script
        if re.search(r"[a-zA-Z]", text):
            return "en"
        return "unknown"
    try:
        return detect(clean)
    except Exception:
        return "unknown"


def is_english(text) -> bool:
    """Return True if the text is in English (or too short to tell but Latin)."""
    lang = detect_language(text)
    return lang == "en"


# ---------------------------------------------------------------------------
# Step 4: Standard NLP preprocessing
# ---------------------------------------------------------------------------

_STOP_WORDS = None
_LEMMATIZER = None


def _get_nlp_tools():
    global _STOP_WORDS, _LEMMATIZER
    if not NLTK_AVAILABLE:
        return None, None
    if _STOP_WORDS is None:
        try:
            _STOP_WORDS = set(stopwords.words("english"))
        except Exception:
            _STOP_WORDS = set()
    if _LEMMATIZER is None:
        try:
            _LEMMATIZER = WordNetLemmatizer()
        except Exception:
            _LEMMATIZER = None
    return _STOP_WORDS, _LEMMATIZER


def clean_text_basic(text) -> str:
    """Basic cleaning: lowercase, remove URLs, mentions, non-ASCII junk."""
    text = _to_str(text)
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)   # URLs
    text = re.sub(r"@\w+", "", text)                # mentions
    text = re.sub(r"#(\w+)", r"\1", text)           # keep hashtag text
    # Remove non-ASCII characters (leftover mojibake, non-Latin scripts)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_for_nlp(text) -> str:
    """NLP preprocessing: tokenise, remove stopwords, lemmatise.

    If NLTK is unavailable or fails, falls back to simple split-based
    processing so we never silently return empty/NaN.
    """
    text = _to_str(text)
    if not text:
        return ""

    stop_words, lemmatizer = _get_nlp_tools()

    # Strip punctuation
    text_clean = text.translate(str.maketrans("", "", string.punctuation))
    text_clean = re.sub(r"\s+", " ", text_clean).strip()

    if not text_clean:
        return ""

    # Try NLTK tokenisation; fall back to simple split
    try:
        tokens = word_tokenize(text_clean)
    except Exception:
        tokens = text_clean.split()

    if stop_words is None:
        stop_words = set()

    processed_tokens = []
    for w in tokens:
        if w in stop_words or len(w) < 2:
            continue
        if lemmatizer is not None:
            try:
                w = lemmatizer.lemmatize(w)
            except Exception:
                pass
        processed_tokens.append(w)

    return " ".join(processed_tokens)


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

def run_cleaning_pipeline(
    df: pd.DataFrame,
    comment_col: str = "Comment",
    progress_callback=None,
) -> pd.DataFrame:
    """Run the complete cleaning pipeline on a comments DataFrame.

    Strategy:
        1. Fix encoding (mojibake from YouTube API)
        2. Remove emojis
        3. Detect language → **keep only English**, drop the rest
        4. Clean remaining English comments (URLs, mentions, non-ASCII junk)
        5. NLP preprocess (stopwords, lemmatisation)
        6. Drop comments that are truly empty after cleaning

    Returns a new DataFrame with additional columns:
        - comment_fixed    : encoding-fixed text
        - comment_no_emoji : emojis removed
        - detected_lang    : ISO 639-1 language code
        - comment_clean    : basic cleaning applied
        - comment_processed: fully preprocessed for NLP
    """
    out = df.copy()

    def _progress(step, frac):
        if progress_callback:
            progress_callback(step, frac)

    # 0. Drop rows where the comment is truly empty / NaN
    _progress("Dropping empty rows", 0.0)
    out = out[out[comment_col].notna()].copy()
    out = out[out[comment_col].astype(str).str.strip().ne("")].copy()
    out = out[~out[comment_col].astype(str).str.strip().str.lower().isin(["nan", "none"])].copy()
    out = out.reset_index(drop=True)
    _progress("Dropping empty rows", 1.0)

    # 1. Fix encoding
    _progress("Fixing encoding", 0.0)
    out["comment_fixed"] = out[comment_col].apply(fix_encoding).fillna("")
    _progress("Fixing encoding", 1.0)

    # 2. Remove emojis
    _progress("Removing emojis", 0.0)
    out["comment_no_emoji"] = out["comment_fixed"].apply(remove_emojis).fillna("")
    _progress("Removing emojis", 1.0)

    # 3. Detect language
    _progress("Detecting languages", 0.0)
    out["detected_lang"] = out["comment_no_emoji"].apply(detect_language).fillna("unknown")
    _progress("Detecting languages", 1.0)

    # 4. Keep only English comments
    _progress("Filtering English only", 0.0)
    n_before = len(out)
    out = out[out["detected_lang"] == "en"].reset_index(drop=True)
    n_dropped_lang = n_before - len(out)
    _progress("Filtering English only", 1.0)

    # 5. Basic text cleaning
    _progress("Cleaning text", 0.0)
    out["comment_clean"] = out["comment_no_emoji"].apply(clean_text_basic).fillna("")
    _progress("Cleaning text", 1.0)

    # 6. NLP preprocessing
    _progress("NLP preprocessing", 0.0)
    out["comment_processed"] = out["comment_clean"].apply(preprocess_for_nlp).fillna("")
    _progress("NLP preprocessing", 1.0)

    # 7. Fallback: if NLP processing emptied a comment but clean text has
    #    content, keep the clean text so we don't lose valid data.
    fallback_mask = (out["comment_processed"].str.strip() == "") & (out["comment_clean"].str.strip() != "")
    out.loc[fallback_mask, "comment_processed"] = out.loc[fallback_mask, "comment_clean"]

    # 8. Drop rows where both clean and processed are truly empty
    empty_mask = (
        out["comment_processed"].isna()
        | (out["comment_processed"].str.strip() == "")
    )
    n_before2 = len(out)
    out = out[~empty_mask].reset_index(drop=True)
    n_dropped_empty = n_before2 - len(out)

    _progress("Done", 1.0)
    return out


def cleaning_report(original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> dict:
    """Produce a summary report of what the cleaning pipeline did."""
    original_count = len(original_df)
    cleaned_count = len(cleaned_df)
    return {
        "original_count": original_count,
        "cleaned_count": cleaned_count,
        "removed_count": original_count - cleaned_count,
        "removal_pct": round(
            (original_count - cleaned_count) / max(original_count, 1) * 100, 1
        ),
        "avg_tokens_after": round(
            cleaned_df["comment_processed"].str.split().str.len().mean(), 1
        ) if len(cleaned_df) > 0 else 0,
    }
