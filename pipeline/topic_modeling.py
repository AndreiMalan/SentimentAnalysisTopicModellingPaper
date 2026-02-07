"""
Topic Modeling Module
=====================
Multiple algorithms for classifying YouTube comments into literature-based
constructs (topics).  All methods return a standardised DataFrame with columns:
    text, brand, assigned_topic, confidence, method

Algorithms implemented:
1. Keyword-based classification (inclusion/exclusion from literature)
2. LDA (Latent Dirichlet Allocation)
3. NMF (Non-negative Matrix Factorisation)
4. Seeded / Guided LDA
5. BERTopic (transformer-based)
6. Zero-Shot Classification (facebook/bart-large-mnli)
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

from config.constructs import CONSTRUCTS, CONSTRUCT_NAMES, RANDOM_STATE

# Optional imports --------------------------------------------------------
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

try:
    from transformers import pipeline as hf_pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ========================================================================
# 1. KEYWORD-BASED CLASSIFICATION
# ========================================================================

class KeywordClassifier:
    """Classify comments using inclusion/exclusion keywords from literature
    constructs.  Fast, fully interpretable, no training needed."""

    def __init__(self, constructs: Dict = None):
        self.constructs = constructs or CONSTRUCTS

    def _score_text(self, text: str) -> Dict[str, float]:
        text_lower = text.lower()
        scores: Dict[str, float] = {}
        for name, info in self.constructs.items():
            inclusion = info["inclusion_keywords"]
            exclusion = info.get("exclusion_keywords", [])

            inc_hits = sum(1 for kw in inclusion if kw.lower() in text_lower)

            # Penalise exclusion matches
            exc_hits = sum(1 for kw in exclusion if kw.lower() in text_lower)

            raw = inc_hits - 0.5 * exc_hits
            scores[name] = max(0.0, raw / max(len(inclusion), 1))
        return scores

    def classify(self, texts: List[str], brands: List[str] = None) -> pd.DataFrame:
        rows = []
        for i, text in enumerate(texts):
            scores = self._score_text(text)
            best = max(scores, key=scores.get)
            conf = scores[best]
            rows.append({
                "text": text,
                "brand": brands[i] if brands else None,
                "assigned_topic": best if conf > 0 else "Unclassified",
                "confidence": conf,
                "method": "Keyword",
                **{f"score_{k}": v for k, v in scores.items()},
            })
        return pd.DataFrame(rows)


# ========================================================================
# 2. LDA
# ========================================================================

class LDATopicModel:
    """Standard Latent Dirichlet Allocation."""

    def __init__(self, n_topics: int = 7, max_features: int = 2000):
        self.n_topics = n_topics
        self.max_features = max_features
        self.vectorizer = None
        self.model = None
        self.topic_mapping: Dict[int, str] = {}
        self.topic_words: Dict[str, List[str]] = {}

    def fit(self, texts: List[str]):
        self.vectorizer = CountVectorizer(
            max_features=self.max_features, max_df=0.95, min_df=2, stop_words="english"
        )
        dtm = self.vectorizer.fit_transform(texts)
        self.model = LatentDirichletAllocation(
            n_components=self.n_topics, random_state=RANDOM_STATE,
            max_iter=30, learning_method="online", n_jobs=-1,
        )
        self.model.fit(dtm)
        self._map_topics()
        return self

    def _map_topics(self, n_top: int = 20):
        feat = self.vectorizer.get_feature_names_out()
        for idx in range(self.n_topics):
            top_ids = self.model.components_[idx].argsort()[-n_top:][::-1]
            words = [feat[i] for i in top_ids]
            self.topic_words[f"LDA_Topic_{idx}"] = words

            best_name, best_overlap = "Unclassified", 0
            word_set = set(words)
            for cname, cinfo in CONSTRUCTS.items():
                kw_set = {k.lower() for k in cinfo["inclusion_keywords"]}
                overlap = len(word_set & kw_set)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_name = cname
            self.topic_mapping[idx] = best_name

    def predict(self, texts: List[str], brands: List[str] = None) -> pd.DataFrame:
        dtm = self.vectorizer.transform(texts)
        doc_topics = self.model.transform(dtm)
        rows = []
        for i, dist in enumerate(doc_topics):
            top_idx = int(np.argmax(dist))
            rows.append({
                "text": texts[i],
                "brand": brands[i] if brands else None,
                "assigned_topic": self.topic_mapping.get(top_idx, "Unclassified"),
                "confidence": float(dist[top_idx]),
                "method": "LDA",
                "lda_topic_id": top_idx,
            })
        return pd.DataFrame(rows)


# ========================================================================
# 3. NMF
# ========================================================================

class NMFTopicModel:
    """Non-negative Matrix Factorisation topic model."""

    def __init__(self, n_topics: int = 7, max_features: int = 2000):
        self.n_topics = n_topics
        self.max_features = max_features
        self.vectorizer = None
        self.model = None
        self.topic_mapping: Dict[int, str] = {}
        self.topic_words: Dict[str, List[str]] = {}

    def fit(self, texts: List[str]):
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features, max_df=0.95, min_df=2, stop_words="english"
        )
        dtm = self.vectorizer.fit_transform(texts)
        self.model = NMF(
            n_components=self.n_topics, random_state=RANDOM_STATE,
            init="nndsvda", max_iter=400,
        )
        self.model.fit(dtm)
        self._map_topics()
        return self

    def _map_topics(self, n_top: int = 20):
        feat = self.vectorizer.get_feature_names_out()
        for idx in range(self.n_topics):
            top_ids = self.model.components_[idx].argsort()[-n_top:][::-1]
            words = [feat[i] for i in top_ids]
            self.topic_words[f"NMF_Topic_{idx}"] = words

            best_name, best_overlap = "Unclassified", 0
            word_set = set(words)
            for cname, cinfo in CONSTRUCTS.items():
                kw_set = {k.lower() for k in cinfo["inclusion_keywords"]}
                overlap = len(word_set & kw_set)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_name = cname
            self.topic_mapping[idx] = best_name

    def predict(self, texts: List[str], brands: List[str] = None) -> pd.DataFrame:
        dtm = self.vectorizer.transform(texts)
        doc_topics = self.model.transform(dtm)
        rows = []
        for i, dist in enumerate(doc_topics):
            top_idx = int(np.argmax(dist))
            norm = dist / (dist.sum() + 1e-10)
            rows.append({
                "text": texts[i],
                "brand": brands[i] if brands else None,
                "assigned_topic": self.topic_mapping.get(top_idx, "Unclassified"),
                "confidence": float(norm[top_idx]),
                "method": "NMF",
                "nmf_topic_id": top_idx,
            })
        return pd.DataFrame(rows)


# ========================================================================
# 4. SEEDED / GUIDED LDA
# ========================================================================

class SeededLDA:
    """LDA with seed-word boosting from the literature constructs."""

    def __init__(self, n_topics: int = 7, max_features: int = 2000, boost_factor: float = 5.0):
        self.n_topics = n_topics
        self.max_features = max_features
        self.boost_factor = boost_factor
        self.vectorizer = None
        self.model = None
        self.topic_names: List[str] = []
        self.topic_words: Dict[str, List[str]] = {}

    def fit(self, texts: List[str]):
        self.vectorizer = CountVectorizer(
            max_features=self.max_features, max_df=0.95, min_df=2, stop_words="english"
        )
        dtm = self.vectorizer.fit_transform(texts)
        feat = self.vectorizer.get_feature_names_out()
        word2idx = {w: i for i, w in enumerate(feat)}

        self.model = LatentDirichletAllocation(
            n_components=self.n_topics, random_state=RANDOM_STATE,
            max_iter=50, learning_method="batch", n_jobs=-1,
        )
        self.model.fit(dtm)

        self.topic_names = list(CONSTRUCTS.keys())[:self.n_topics]
        for t_idx, cname in enumerate(self.topic_names):
            if t_idx >= self.n_topics:
                break
            for kw in CONSTRUCTS[cname]["inclusion_keywords"]:
                kw_lower = kw.lower()
                if kw_lower in word2idx:
                    self.model.components_[t_idx, word2idx[kw_lower]] *= self.boost_factor

        # Renormalise
        self.model.components_ /= self.model.components_.sum(axis=1, keepdims=True)

        # Extract top words
        for t_idx in range(self.n_topics):
            top_ids = self.model.components_[t_idx].argsort()[-15:][::-1]
            words = [feat[i] for i in top_ids]
            name = self.topic_names[t_idx] if t_idx < len(self.topic_names) else f"Topic_{t_idx}"
            self.topic_words[name] = words

        return self

    def predict(self, texts: List[str], brands: List[str] = None) -> pd.DataFrame:
        dtm = self.vectorizer.transform(texts)
        doc_topics = self.model.transform(dtm)
        rows = []
        for i, dist in enumerate(doc_topics):
            top_idx = int(np.argmax(dist))
            name = self.topic_names[top_idx] if top_idx < len(self.topic_names) else "Unclassified"
            rows.append({
                "text": texts[i],
                "brand": brands[i] if brands else None,
                "assigned_topic": name,
                "confidence": float(dist[top_idx]),
                "method": "Seeded LDA",
            })
        return pd.DataFrame(rows)


# ========================================================================
# 5. BERTopic
# ========================================================================

class BERTopicModel:
    """BERTopic with optional seed-topic guidance."""

    def __init__(self, n_topics: int = 7, use_seeds: bool = True):
        if not BERTOPIC_AVAILABLE:
            raise ImportError("pip install bertopic sentence-transformers")
        self.n_topics = n_topics
        self.use_seeds = use_seeds
        self.model = None
        self.topic_mapping: Dict[int, str] = {}
        self.topic_words: Dict[str, List[str]] = {}

    def fit(self, texts: List[str]):
        emb = SentenceTransformer("all-MiniLM-L6-v2")
        seed_list = None
        if self.use_seeds:
            seed_list = [
                info["inclusion_keywords"][:10]
                for info in CONSTRUCTS.values()
            ]
        self.model = BERTopic(
            embedding_model=emb,
            min_topic_size=max(10, len(texts) // 100),
            nr_topics=self.n_topics,
            seed_topic_list=seed_list,
            calculate_probabilities=True,
            verbose=False,
        )
        self.model.fit_transform(texts)
        self._map_topics()
        return self

    def _map_topics(self):
        info = self.model.get_topic_info()
        cnames = list(CONSTRUCTS.keys())
        for _, row in info.iterrows():
            tid = row["Topic"]
            if tid == -1:
                self.topic_mapping[tid] = "Unclassified"
                continue
            tw = self.model.get_topic(tid)
            if not tw:
                self.topic_mapping[tid] = "Unclassified"
                continue
            words = [w for w, _ in tw[:15]]
            self.topic_words[f"BERTopic_{tid}"] = words

            best_name, best_ov = "Unclassified", 0
            word_set = set(words)
            for cn, ci in CONSTRUCTS.items():
                kw_set = {k.lower() for k in ci["inclusion_keywords"]}
                ov = len(word_set & kw_set)
                if ov > best_ov:
                    best_ov = ov
                    best_name = cn
            self.topic_mapping[tid] = best_name

    def predict(self, texts: List[str], brands: List[str] = None) -> pd.DataFrame:
        topics, probs = self.model.transform(texts)
        rows = []
        for i, (tid, prob) in enumerate(zip(topics, probs)):
            conf = float(np.max(prob)) if prob is not None and len(prob) > 0 else 0.0
            rows.append({
                "text": texts[i],
                "brand": brands[i] if brands else None,
                "assigned_topic": self.topic_mapping.get(tid, "Unclassified"),
                "confidence": conf,
                "method": "BERTopic",
                "bertopic_id": tid,
            })
        return pd.DataFrame(rows)


# ========================================================================
# 6. ZERO-SHOT CLASSIFICATION
# ========================================================================

class ZeroShotClassifier:
    """Zero-shot classification using facebook/bart-large-mnli."""

    def __init__(self):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("pip install transformers torch")
        device = 0 if torch.cuda.is_available() else -1
        self.classifier = hf_pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=device,
        )
        self.labels = CONSTRUCT_NAMES

    def classify(
        self, texts: List[str], brands: List[str] = None,
        progress_callback=None,
    ) -> pd.DataFrame:
        rows = []
        total = len(texts)
        for i, text in enumerate(texts):
            try:
                res = self.classifier(
                    text[:512], candidate_labels=self.labels, multi_label=False
                )
                rows.append({
                    "text": text,
                    "brand": brands[i] if brands else None,
                    "assigned_topic": res["labels"][0],
                    "confidence": float(res["scores"][0]),
                    "method": "Zero-Shot",
                    **{f"score_{l}": s for l, s in zip(res["labels"], res["scores"])},
                })
            except Exception:
                rows.append({
                    "text": text,
                    "brand": brands[i] if brands else None,
                    "assigned_topic": "Error",
                    "confidence": 0.0,
                    "method": "Zero-Shot",
                })
            if progress_callback and i % 10 == 0:
                progress_callback(i / total)
        return pd.DataFrame(rows)


# ========================================================================
# UTILITY: run all algorithms and return combined results
# ========================================================================

def run_all_topic_models(
    texts: List[str],
    brands: List[str] = None,
    methods: List[str] = None,
    progress_callback=None,
) -> Dict[str, pd.DataFrame]:
    """Run selected (or all) topic-modeling methods and return dict of results.

    Parameters
    ----------
    methods : list of str, optional
        Subset of ["Keyword", "LDA", "NMF", "Seeded LDA", "BERTopic", "Zero-Shot"].
        Defaults to all available methods.
    """
    available = {
        "Keyword": True,
        "LDA": True,
        "NMF": True,
        "Seeded LDA": True,
        "BERTopic": BERTOPIC_AVAILABLE,
        "Zero-Shot": TRANSFORMERS_AVAILABLE,
    }
    if methods is None:
        methods = [m for m, ok in available.items() if ok]

    results: Dict[str, pd.DataFrame] = {}

    if "Keyword" in methods:
        if progress_callback:
            progress_callback("Keyword classification", 0.0)
        kw = KeywordClassifier()
        results["Keyword"] = kw.classify(texts, brands)
        if progress_callback:
            progress_callback("Keyword classification", 1.0)

    if "LDA" in methods:
        if progress_callback:
            progress_callback("LDA", 0.0)
        lda = LDATopicModel(n_topics=len(CONSTRUCTS))
        lda.fit(texts)
        results["LDA"] = lda.predict(texts, brands)
        results["LDA_model"] = lda
        if progress_callback:
            progress_callback("LDA", 1.0)

    if "NMF" in methods:
        if progress_callback:
            progress_callback("NMF", 0.0)
        nmf = NMFTopicModel(n_topics=len(CONSTRUCTS))
        nmf.fit(texts)
        results["NMF"] = nmf.predict(texts, brands)
        results["NMF_model"] = nmf
        if progress_callback:
            progress_callback("NMF", 1.0)

    if "Seeded LDA" in methods:
        if progress_callback:
            progress_callback("Seeded LDA", 0.0)
        slda = SeededLDA(n_topics=len(CONSTRUCTS))
        slda.fit(texts)
        results["Seeded LDA"] = slda.predict(texts, brands)
        results["Seeded LDA_model"] = slda
        if progress_callback:
            progress_callback("Seeded LDA", 1.0)

    if "BERTopic" in methods and BERTOPIC_AVAILABLE:
        if progress_callback:
            progress_callback("BERTopic", 0.0)
        bt = BERTopicModel(n_topics=len(CONSTRUCTS))
        bt.fit(texts)
        results["BERTopic"] = bt.predict(texts, brands)
        results["BERTopic_model"] = bt
        if progress_callback:
            progress_callback("BERTopic", 1.0)

    if "Zero-Shot" in methods and TRANSFORMERS_AVAILABLE:
        if progress_callback:
            progress_callback("Zero-Shot", 0.0)
        zs = ZeroShotClassifier()
        results["Zero-Shot"] = zs.classify(texts, brands, progress_callback=progress_callback)
        if progress_callback:
            progress_callback("Zero-Shot", 1.0)

    return results
