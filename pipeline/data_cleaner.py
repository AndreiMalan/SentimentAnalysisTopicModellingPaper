"""
Data Cleaning Pipeline
======================
Handles the full text-cleaning workflow for YouTube comments:
1. Fix encoding / mojibake (UTF-8 emoji artifacts)
2. Remove or normalize emoji characters
3. Detect language
4. Translate non-English comments to English
5. Standard NLP preprocessing (lowercase, URLs, punctuation, stopwords, lemmatization)
6. Filter short / empty comments
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
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False

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
# Step 1: Fix encoding / mojibake
# ---------------------------------------------------------------------------

def fix_encoding(text: str) -> str:
    """Fix mojibake caused by YouTube API encoding issues."""
    if not isinstance(text, str):
        return str(text)
    if FTFY_AVAILABLE:
        return ftfy.fix_text(text)
    return text


# ---------------------------------------------------------------------------
# Step 2: Remove / normalise emojis
# ---------------------------------------------------------------------------

def remove_emojis(text: str) -> str:
    """Remove emoji characters from text.

    Uses the ``emoji`` library which has a curated list of actual emoji
    code-points.  A manual regex fallback is provided for when the library
    is not installed, but it is intentionally narrow to avoid accidentally
    stripping CJK, Korean, Arabic, or other legitimate scripts.
    """
    if EMOJI_AVAILABLE:
        text = emoji.replace_emoji(text, replace="")
    else:
        # Narrow fallback: only the most common emoji blocks, nothing that
        # overlaps CJK / Hangul / Arabic / Devanagari.
        _fallback = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U0001F900-\U0001F9FF"  # supplemental symbols
            "\U0001FA00-\U0001FA6F"  # chess symbols
            "\U0001FA70-\U0001FAFF"  # symbols extended-A
            "\u200d"                 # zero-width joiner
            "\ufe0f"                 # variation selector
            "]+",
            flags=re.UNICODE,
        )
        text = _fallback.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Step 3: Language detection
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """Detect language of a text string. Returns ISO 639-1 code or 'unknown'."""
    if not LANGDETECT_AVAILABLE:
        return "unknown"
    clean = re.sub(r"[^a-zA-Z\u0400-\u04FF\u0600-\u06FF\u0900-\u097F\u4e00-\u9fff\s]", "", text)
    if len(clean.split()) < 3:
        return "en"  # too short to detect reliably; assume English
    try:
        return detect(clean)
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Step 4: Translation
# ---------------------------------------------------------------------------

def translate_to_english(text: str, source_lang: str = "auto") -> str:
    """Translate text to English using Google Translate (free tier)."""
    if not TRANSLATOR_AVAILABLE:
        return text
    if not text or len(text.strip()) < 3:
        return text
    try:
        translated = GoogleTranslator(source=source_lang, target="en").translate(text)
        return translated if translated else text
    except Exception:
        return text


def batch_translate(texts: list, source_langs: list = None) -> list:
    """Translate a list of texts, skipping already-English ones."""
    results = []
    for i, text in enumerate(texts):
        lang = source_langs[i] if source_langs else "auto"
        if lang == "en":
            results.append(text)
        else:
            results.append(translate_to_english(text, source_lang="auto"))
    return results


# ---------------------------------------------------------------------------
# Step 5: Standard NLP preprocessing
# ---------------------------------------------------------------------------

_STOP_WORDS = None
_LEMMATIZER = None


def _get_nlp_tools():
    global _STOP_WORDS, _LEMMATIZER
    if not NLTK_AVAILABLE:
        raise ImportError(
            "nltk is required for preprocessing. "
            "Install with: pip install nltk"
        )
    if _STOP_WORDS is None:
        _STOP_WORDS = set(stopwords.words("english"))
    if _LEMMATIZER is None:
        _LEMMATIZER = WordNetLemmatizer()
    return _STOP_WORDS, _LEMMATIZER


def clean_text_basic(text: str) -> str:
    """Basic cleaning: lowercase, remove URLs, mentions, hashtags.

    Numbers and punctuation are kept at this stage so that downstream
    keyword matching (e.g. "5g", "usb 3.1", "4k") still works.
    They are only stripped later in the NLP-preprocessing step.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)   # URLs
    text = re.sub(r"@\w+", "", text)                # mentions
    text = re.sub(r"#(\w+)", r"\1", text)           # keep hashtag text
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_for_nlp(text: str) -> str:
    """NLP preprocessing: tokenise, remove stopwords, lemmatise.

    Keeps tokens of length >= 2 so short but meaningful words
    (e.g. 'ok', 'no', '5g', 'tv') are preserved.
    """
    stop_words, lemmatizer = _get_nlp_tools()
    # Strip punctuation only for the NLP-processed column
    text_clean = text.translate(str.maketrans("", "", string.punctuation))
    text_clean = re.sub(r"\s+", " ", text_clean).strip()
    tokens = word_tokenize(text_clean)
    tokens = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in stop_words and len(w) >= 2
    ]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

def _is_truly_empty(text: str) -> bool:
    """Return True only if the text has no meaningful content at all.

    Checks for ANY Unicode letter (Latin, Cyrillic, Devanagari, CJK, Arabic,
    etc.) — not just a-zA-Z.  A comment in Hindi or Chinese is NOT empty.
    """
    # \w matches [a-zA-Z0-9_] plus Unicode letters/digits
    # We check for at least one Unicode letter category character
    return not bool(re.search(r"[^\W\d_]", str(text), re.UNICODE))


def run_cleaning_pipeline(
    df: pd.DataFrame,
    comment_col: str = "Comment",
    translate: bool = True,
    min_token_length: int = 1,
    progress_callback=None,
) -> pd.DataFrame:
    """Run the complete cleaning pipeline on a comments DataFrame.

    Only removes rows that are truly empty (no alphabetic characters at all
    after encoding fix, emoji removal, and translation).  Short but
    meaningful comments like "Waiting", "Launch date" are kept.

    Returns a new DataFrame with additional columns:
        - comment_fixed   : encoding-fixed text
        - comment_no_emoji: emojis removed
        - detected_lang   : ISO 639-1 language code
        - comment_english : translated to English (if needed)
        - comment_clean   : basic cleaning applied
        - comment_processed: fully preprocessed for NLP
        - is_english      : bool flag

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *comment_col* column.
    translate : bool
        Whether to translate non-English comments.
    min_token_length : int
        Minimum number of tokens after NLP preprocessing; comments with
        fewer tokens are dropped.  Default is 1 (only drop completely
        empty comments).
    progress_callback : callable or None
        Called with (step_name: str, fraction: float) for progress tracking.
    """
    out = df.copy()
    total = len(out)

    def _progress(step, frac):
        if progress_callback:
            progress_callback(step, frac)

    # 1. Fix encoding
    _progress("Fixing encoding", 0.0)
    out["comment_fixed"] = out[comment_col].apply(fix_encoding)
    _progress("Fixing encoding", 1.0)

    # 2. Remove emojis
    _progress("Removing emojis", 0.0)
    out["comment_no_emoji"] = out["comment_fixed"].apply(remove_emojis)
    _progress("Removing emojis", 1.0)

    # 3. THE ONLY HARD DROP — remove rows that have no Unicode letter in
    #    ANY script (Latin, Cyrillic, Devanagari, CJK, Arabic …) after
    #    encoding fix + emoji removal.  These are truly unrecoverable
    #    (pure emoji strings, bare punctuation, empty strings).
    _progress("Removing empty rows", 0.0)
    empty_mask = out["comment_no_emoji"].apply(_is_truly_empty)
    out = out[~empty_mask].reset_index(drop=True)
    _progress("Removing empty rows", 1.0)

    # 4. Language detection
    _progress("Detecting languages", 0.0)
    out["detected_lang"] = out["comment_no_emoji"].apply(detect_language)
    _progress("Detecting languages", 1.0)

    out["is_english"] = out["detected_lang"] == "en"

    # 5. Translation — translate non-English comments; keep originals on failure
    if translate and TRANSLATOR_AVAILABLE:
        _progress("Translating non-English comments", 0.0)
        non_en_mask = ~out["is_english"]
        non_en_texts = out.loc[non_en_mask, "comment_no_emoji"].tolist()
        if non_en_texts:
            translated = []
            batch_size = 50
            for i in range(0, len(non_en_texts), batch_size):
                batch = non_en_texts[i : i + batch_size]
                translated.extend(batch_translate(batch))
                _progress(
                    "Translating non-English comments",
                    min(1.0, (i + batch_size) / len(non_en_texts)),
                )
            out.loc[non_en_mask, "comment_english"] = translated
        out["comment_english"] = out["comment_english"].fillna(out["comment_no_emoji"])
    else:
        out["comment_english"] = out["comment_no_emoji"]
    _progress("Translating non-English comments", 1.0)

    # 6. Basic text cleaning (keeps numbers and meaningful short words)
    _progress("Cleaning text", 0.0)
    out["comment_clean"] = out["comment_english"].apply(clean_text_basic)
    _progress("Cleaning text", 1.0)

    # 7. NLP preprocessing
    _progress("NLP preprocessing", 0.0)
    out["comment_processed"] = out["comment_clean"].apply(preprocess_for_nlp)
    _progress("NLP preprocessing", 1.0)

    # 8. If NLP preprocessing emptied a comment (e.g. non-Latin text that
    #    the English lemmatizer couldn't handle, or all-stopword comments),
    #    fall back to comment_clean so the row is NOT lost.
    fallback_mask = out["comment_processed"].str.strip() == ""
    out.loc[fallback_mask, "comment_processed"] = out.loc[
        fallback_mask, "comment_clean"
    ]

    _progress("Done", 1.0)
    return out


def cleaning_report(original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> dict:
    """Produce a summary report of what the cleaning pipeline did."""
    original_count = len(original_df)
    cleaned_count = len(cleaned_df)
    lang_dist = cleaned_df["detected_lang"].value_counts().to_dict() if "detected_lang" in cleaned_df.columns else {}
    return {
        "original_count": original_count,
        "cleaned_count": cleaned_count,
        "removed_count": original_count - cleaned_count,
        "removal_pct": round((original_count - cleaned_count) / original_count * 100, 1),
        "english_pct": round(
            cleaned_df["is_english"].sum() / cleaned_count * 100, 1
        ) if "is_english" in cleaned_df.columns else None,
        "language_distribution": lang_dist,
        "avg_tokens_after": round(
            cleaned_df["comment_processed"].str.split().str.len().mean(), 1
        ),
    }
