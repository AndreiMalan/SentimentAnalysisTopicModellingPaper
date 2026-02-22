"""
Topic Modeling Module
=====================
Three algorithms for classifying YouTube comments into topics:

1. LDA  — Fully **unsupervised**.  Discovers topics on its own.
   Tests k = k_min … k_max, evaluates each with perplexity, coherence
   (UMass), and log-likelihood.  Selects optimal k automatically.
   Produces LDAvis-style word lists per topic.

2. Seeded LDA  — **Semi-supervised**.  Guided by literature-construct
   seed words.  Maps discovered topics to the 7 predefined constructs.

3. BERTopic  — **Transformer-based**, semi-supervised.  Semantic
   clustering with seed topic guidance, mapped to constructs.

All predict() methods return a standardised DataFrame with columns:
    text, brand, assigned_topic, confidence, method
"""

import re
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from config.constructs import CONSTRUCTS, CONSTRUCT_NAMES, RANDOM_STATE

# Optional imports --------------------------------------------------------
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False


# ========================================================================
# HELPERS — COHERENCE METRICS
# ========================================================================

def _coherence_umass(model, dtm, n_top_words=10):
    """UMass coherence (Mimno et al., 2011).

    C_UMass = (2 / T(T-1)) * Σ_{i<j} log( (D(w_i, w_j) + 1) / D(w_i) )

    Values are negative; closer to 0 is better.
    """
    dtm_bin = (dtm > 0).astype(int)
    scores = []
    for k in range(model.n_components):
        top_ids = model.components_[k].argsort()[-n_top_words:][::-1]
        topic_score = 0.0
        n_pairs = 0
        for i in range(1, len(top_ids)):
            for j in range(i):
                wi, wj = top_ids[i], top_ids[j]
                d_wj = int(dtm_bin[:, wj].sum())
                d_wi_wj = int(dtm_bin[:, wi].multiply(dtm_bin[:, wj]).sum())
                if d_wj > 0:
                    topic_score += np.log((d_wi_wj + 1) / d_wj)
                n_pairs += 1
        if n_pairs > 0:
            scores.append(topic_score / n_pairs)
    return float(np.mean(scores)) if scores else 0.0


def _coherence_npmi(model, dtm, n_top_words=10):
    """NPMI coherence (Bouma, 2009).

    NPMI(w_i, w_j) = ( log P(w_i,w_j)/(P(w_i)P(w_j)) ) / ( -log P(w_i,w_j) )
    Normalised to [-1, 1].  Higher is better.
    """
    n_docs = dtm.shape[0]
    dtm_bin = (dtm > 0).astype(int)
    scores = []
    for k in range(model.n_components):
        top_ids = model.components_[k].argsort()[-n_top_words:][::-1]
        topic_score = 0.0
        n_pairs = 0
        for i in range(1, len(top_ids)):
            for j in range(i):
                wi, wj = top_ids[i], top_ids[j]
                p_wi = int(dtm_bin[:, wi].sum()) / n_docs
                p_wj = int(dtm_bin[:, wj].sum()) / n_docs
                p_wi_wj = int(dtm_bin[:, wi].multiply(dtm_bin[:, wj]).sum()) / n_docs
                if p_wi_wj > 0 and p_wi > 0 and p_wj > 0:
                    pmi = np.log(p_wi_wj / (p_wi * p_wj))
                    npmi = pmi / (-np.log(p_wi_wj))
                    topic_score += npmi
                else:
                    topic_score += -1.0
                n_pairs += 1
        if n_pairs > 0:
            scores.append(topic_score / n_pairs)
    return float(np.mean(scores)) if scores else 0.0


def _topic_diversity(model, vectorizer, n_top_words=25):
    """Proportion of unique words across all topics' top-N (Dieng et al., 2020)."""
    feat = vectorizer.get_feature_names_out()
    all_words = []
    for k in range(model.n_components):
        top_ids = model.components_[k].argsort()[-n_top_words:][::-1]
        all_words.extend([feat[i] for i in top_ids])
    return len(set(all_words)) / len(all_words) if all_words else 0.0


# ========================================================================
# 1. LDA — FULLY UNSUPERVISED
# ========================================================================

class LDATopicModel:
    """Unsupervised LDA with automatic k-selection.

    Trains one model per k in [k_min, k_max] and evaluates each with
    perplexity, UMass coherence, NPMI coherence, log-likelihood, and
    topic diversity.  Selects the k with the best coherence score.
    """

    def __init__(self, k_min=4, k_max=15, max_features=2000, n_top_words=20):
        self.k_min = k_min
        self.k_max = k_max
        self.max_features = max_features
        self.n_top_words = n_top_words

        # Fitted state
        self.vectorizer = None
        self.dtm = None
        self.models: Dict[int, LatentDirichletAllocation] = {}
        self.perplexity_scores: Dict[int, float] = {}
        self.coherence_umass: Dict[int, float] = {}
        self.coherence_npmi: Dict[int, float] = {}
        self.log_likelihood: Dict[int, float] = {}
        self.diversity_scores: Dict[int, float] = {}
        self.optimal_k: int = None
        self.best_model = None
        self.topic_words: Dict[str, List[Tuple[str, float]]] = {}
        self.topic_proportions: np.ndarray = None
        self.overall_diversity: float = None

    # ----- fitting -----
    def fit(self, texts: List[str], progress_callback=None):
        self.vectorizer = CountVectorizer(
            max_features=self.max_features, max_df=0.95, min_df=2,
            stop_words="english",
        )
        self.dtm = self.vectorizer.fit_transform(texts)

        k_values = list(range(self.k_min, self.k_max + 1))
        for idx, k in enumerate(k_values):
            if progress_callback:
                progress_callback("LDA", idx / len(k_values))

            model = LatentDirichletAllocation(
                n_components=k, random_state=RANDOM_STATE,
                max_iter=30, learning_method="online", n_jobs=-1,
            )
            model.fit(self.dtm)

            self.models[k] = model
            self.perplexity_scores[k] = model.perplexity(self.dtm)
            self.log_likelihood[k] = model.score(self.dtm)
            self.coherence_umass[k] = _coherence_umass(model, self.dtm)
            self.coherence_npmi[k] = _coherence_npmi(model, self.dtm)
            self.diversity_scores[k] = _topic_diversity(model, self.vectorizer)

        # Optimal k = highest UMass coherence (closest to 0)
        self.optimal_k = max(self.coherence_umass, key=self.coherence_umass.get)
        self.best_model = self.models[self.optimal_k]

        self._extract_topic_info()
        self._print_report()
        return self

    # ----- topic extraction -----
    def _extract_topic_info(self):
        feat = self.vectorizer.get_feature_names_out()
        model = self.best_model
        # Normalise to probability distributions
        tw_dist = model.components_ / model.components_.sum(axis=1, keepdims=True)

        self.topic_words = {}
        for k in range(model.n_components):
            top_ids = tw_dist[k].argsort()[-self.n_top_words:][::-1]
            self.topic_words[f"Topic {k + 1}"] = [
                (feat[i], float(tw_dist[k, i])) for i in top_ids
            ]

        doc_topics = model.transform(self.dtm)
        self.topic_proportions = doc_topics.mean(axis=0)
        self.overall_diversity = _topic_diversity(model, self.vectorizer)

    # ----- prediction -----
    def predict(self, texts: List[str], brands: List[str] = None) -> pd.DataFrame:
        dtm = self.vectorizer.transform(texts)
        doc_topics = self.best_model.transform(dtm)
        rows = []
        for i, dist in enumerate(doc_topics):
            top_idx = int(np.argmax(dist))
            rows.append({
                "text": texts[i],
                "brand": brands[i] if brands else None,
                "assigned_topic": f"Topic {top_idx + 1}",
                "confidence": float(dist[top_idx]),
                "method": "LDA",
                "lda_topic_id": top_idx,
            })
        return pd.DataFrame(rows)

    # ----- k-selection table -----
    def get_k_metrics_df(self) -> pd.DataFrame:
        rows = []
        for k in sorted(self.perplexity_scores.keys()):
            rows.append({
                "k": k,
                "Perplexity": round(self.perplexity_scores[k], 2),
                "Coherence (UMass)": round(self.coherence_umass[k], 4),
                "Coherence (NPMI)": round(self.coherence_npmi[k], 4),
                "Log-Likelihood": round(self.log_likelihood[k], 2),
                "Topic Diversity": round(self.diversity_scores[k], 4),
            })
        df = pd.DataFrame(rows)
        return df

    # ----- console report -----
    def _print_report(self):
        sep = "=" * 90
        print(f"\n{sep}")
        print("LDA TOPIC MODEL — DETAILED METRICS REPORT")
        print(sep)
        print(f"K range tested           : {self.k_min} – {self.k_max}")
        print(f"Optimal k (best UMass)   : {self.optimal_k}")
        print(f"Vocabulary size          : {len(self.vectorizer.get_feature_names_out())}")
        print(f"Documents                : {self.dtm.shape[0]}")

        print(f"\n{'k':>4} | {'Perplexity':>12} | {'C_UMass':>10} | {'C_NPMI':>10} | {'Log-Lik':>14} | {'Diversity':>10}")
        print("-" * 72)
        for k in sorted(self.perplexity_scores.keys()):
            mark = " <-- best" if k == self.optimal_k else ""
            print(
                f"{k:>4d} | "
                f"{self.perplexity_scores[k]:>12.2f} | "
                f"{self.coherence_umass[k]:>10.4f} | "
                f"{self.coherence_npmi[k]:>10.4f} | "
                f"{self.log_likelihood[k]:>14.2f} | "
                f"{self.diversity_scores[k]:>10.4f}{mark}"
            )

        print(f"\n--- Discovered Topics (k = {self.optimal_k}) ---")
        for topic_name, words_weights in self.topic_words.items():
            idx = int(topic_name.split()[-1]) - 1
            prop = self.topic_proportions[idx]
            print(f"\n  {topic_name}  (proportion = {prop:.4f})")
            for word, weight in words_weights[:15]:
                bar = "█" * max(1, int(weight * 1500))
                print(f"    {word:20s} {weight:.6f}  {bar}")

        print(f"\n  Topic Diversity (top-25) : {self.overall_diversity:.4f}")
        print(f"  Best Coherence (UMass)  : {self.coherence_umass[self.optimal_k]:.4f}")
        print(f"  Best Coherence (NPMI)   : {self.coherence_npmi[self.optimal_k]:.4f}")
        print(f"  Perplexity @ optimal k  : {self.perplexity_scores[self.optimal_k]:.2f}")
        print(f"{sep}\n")
        sys.stdout.flush()


# ========================================================================
# 2. SEEDED / GUIDED LDA  (semi-supervised)
# ========================================================================

class SeededLDA:
    """LDA with seed-word initialisation from the literature constructs.

    Unlike plain LDA, this model **initialises** the topic-word
    distributions with high seed-word priors BEFORE fitting, then
    refines via partial_fit.  This ensures construct keywords
    dominate each topic instead of generic corpus words.
    """

    # Brand / product terms to filter from displayed top-words
    _GENERIC_FILTER = {
        "samsung", "xiaomi", "huawei", "apple", "iphone", "galaxy",
        "redmi", "note", "ultra", "pro", "max", "plus", "lite",
        "phone", "mobile", "device", "tab", "tablet", "watch",
        "series", "model", "version", "company", "brand",
    }

    def __init__(self, n_topics: int = 7, max_features: int = 2000,
                 boost_factor: float = 200.0, refine_iters: int = 15):
        self.n_topics = n_topics
        self.max_features = max_features
        self.boost_factor = boost_factor
        self.refine_iters = refine_iters
        self.vectorizer = None
        self.model = None
        self.dtm = None
        self.topic_names: List[str] = []
        self.topic_words: Dict[str, List[Tuple[str, float]]] = {}
        self.perplexity: float = None
        self.coherence_umass_val: float = None
        self.coherence_npmi_val: float = None
        self.topic_diversity_val: float = None

    def fit(self, texts: List[str]):
        self.vectorizer = CountVectorizer(
            max_features=self.max_features, max_df=0.95, min_df=2,
            stop_words="english",
        )
        self.dtm = self.vectorizer.fit_transform(texts)
        feat = self.vectorizer.get_feature_names_out()
        n_features = len(feat)
        word2idx = {w: i for i, w in enumerate(feat)}

        # 1. Quick initial fit to set up internal state
        self.model = LatentDirichletAllocation(
            n_components=self.n_topics, random_state=RANDOM_STATE,
            max_iter=5, learning_method="online", n_jobs=-1,
        )
        self.model.fit(self.dtm)

        # 2. Seed the components BEFORE main training:
        #    Set seed-word positions to a very high value so they
        #    dominate each topic's word distribution.
        self.topic_names = list(CONSTRUCTS.keys())[:self.n_topics]
        for t_idx, cname in enumerate(self.topic_names):
            if t_idx >= self.n_topics:
                break
            for kw in CONSTRUCTS[cname]["inclusion_keywords"]:
                kw_lower = kw.lower()
                if kw_lower in word2idx:
                    self.model.components_[t_idx, word2idx[kw_lower]] *= self.boost_factor

        # 3. Refine with partial_fit: the model updates from the
        #    seed-initialised state so seeds stay prominent.
        for _ in range(self.refine_iters):
            self.model.partial_fit(self.dtm)

        self.model.components_ /= self.model.components_.sum(axis=1, keepdims=True)

        # 4. Extract topic words — filter out generic brand/product terms
        tw_dist = self.model.components_ / self.model.components_.sum(axis=1, keepdims=True)
        for t_idx in range(self.n_topics):
            top_ids = tw_dist[t_idx].argsort()[-50:][::-1]  # get extra to survive filter
            name = self.topic_names[t_idx] if t_idx < len(self.topic_names) else f"Topic_{t_idx}"
            filtered = []
            for idx in top_ids:
                word = feat[idx]
                if word not in self._GENERIC_FILTER:
                    filtered.append((word, float(tw_dist[t_idx, idx])))
                if len(filtered) >= 20:
                    break
            self.topic_words[name] = filtered

        # Metrics
        self.perplexity = self.model.perplexity(self.dtm)
        self.coherence_umass_val = _coherence_umass(self.model, self.dtm)
        self.coherence_npmi_val = _coherence_npmi(self.model, self.dtm)
        self.topic_diversity_val = _topic_diversity(self.model, self.vectorizer)

        self._print_report()
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

    def _print_report(self):
        sep = "=" * 90
        print(f"\n{sep}")
        print("SEEDED LDA — METRICS REPORT")
        print(sep)
        print(f"  Number of topics (constructs) : {self.n_topics}")
        print(f"  Boost factor                  : {self.boost_factor}")
        print(f"  Perplexity                    : {self.perplexity:.2f}")
        print(f"  Coherence (UMass)             : {self.coherence_umass_val:.4f}")
        print(f"  Coherence (NPMI)              : {self.coherence_npmi_val:.4f}")
        print(f"  Topic Diversity               : {self.topic_diversity_val:.4f}")
        print(f"\n--- Construct-Guided Topics ---")
        for tname, words_weights in self.topic_words.items():
            print(f"\n  {tname}:")
            for word, weight in words_weights[:12]:
                print(f"    {word:20s} {weight:.6f}")
        print(f"{sep}\n")
        sys.stdout.flush()


# ========================================================================
# 3. BERTopic  (transformer-based, semi-supervised)
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
        self.topic_words: Dict[str, List[Tuple[str, float]]] = {}

    def fit(self, texts: List[str]):
        emb = SentenceTransformer("all-MiniLM-L6-v2")
        seed_list = None
        if self.use_seeds:
            seed_list = [
                info["inclusion_keywords"][:10]
                for info in CONSTRUCTS.values()
            ]
        # Lower min_topic_size to reduce outliers
        self.model = BERTopic(
            embedding_model=emb,
            min_topic_size=max(5, len(texts) // 200),
            nr_topics=self.n_topics,
            seed_topic_list=seed_list,
            calculate_probabilities=True,
            verbose=False,
        )
        topics, probs = self.model.fit_transform(texts)

        # Reassign outliers (-1) to their nearest topic
        try:
            new_topics = self.model.reduce_outliers(texts, topics, strategy="probabilities")
            self.model.update_topics(texts, topics=new_topics)
        except Exception:
            try:
                new_topics = self.model.reduce_outliers(texts, topics, strategy="distributions")
                self.model.update_topics(texts, topics=new_topics)
            except Exception:
                pass  # keep original assignments if reduce_outliers fails

        self._map_topics()
        self._print_report()
        return self

    def _map_topics(self):
        info = self.model.get_topic_info()
        for _, row in info.iterrows():
            tid = row["Topic"]
            if tid == -1:
                self.topic_mapping[tid] = "Unclassified"
                continue
            tw = self.model.get_topic(tid)
            if not tw:
                self.topic_mapping[tid] = "Unclassified"
                continue
            words = [(w, float(s)) for w, s in tw[:20]]
            self.topic_words[f"BERTopic_{tid}"] = words

            best_name, best_ov = "Unclassified", 0
            word_set = {w for w, _ in words}
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

    def _print_report(self):
        sep = "=" * 90
        print(f"\n{sep}")
        print("BERTopic — METRICS REPORT")
        print(sep)
        n_topics = len([t for t in self.topic_mapping if t != -1])
        print(f"  Discovered topics : {n_topics}")
        print(f"  Seed guidance     : {'Yes' if self.use_seeds else 'No'}")
        print(f"\n--- Topic-Construct Mapping & Words ---")
        for tname, words in self.topic_words.items():
            construct = self.topic_mapping.get(int(tname.split("_")[-1]), "?")
            print(f"\n  {tname} → {construct}")
            for word, weight in words[:10]:
                print(f"    {word:20s} {weight:.4f}")
        print(f"{sep}\n")
        sys.stdout.flush()


# ========================================================================
# UTILITY: run selected algorithms
# ========================================================================

def run_all_topic_models(
    texts: List[str],
    brands: List[str] = None,
    methods: List[str] = None,
    lda_k_min: int = 4,
    lda_k_max: int = 15,
    progress_callback=None,
) -> Dict[str, object]:
    """Run selected topic-modeling methods.

    Parameters
    ----------
    methods : list of str
        Subset of ["LDA", "Seeded LDA", "BERTopic"].
    lda_k_min, lda_k_max : int
        Range of k values to test for unsupervised LDA.
    """
    available = {
        "LDA": True,
        "Seeded LDA": True,
        "BERTopic": BERTOPIC_AVAILABLE,
    }
    if methods is None:
        methods = [m for m, ok in available.items() if ok]

    results: Dict[str, object] = {}

    if "LDA" in methods:
        if progress_callback:
            progress_callback("LDA", 0.0)
        lda = LDATopicModel(k_min=lda_k_min, k_max=lda_k_max)
        lda.fit(texts, progress_callback=progress_callback)
        results["LDA"] = lda.predict(texts, brands)
        results["LDA_model"] = lda
        if progress_callback:
            progress_callback("LDA", 1.0)

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

    return results
