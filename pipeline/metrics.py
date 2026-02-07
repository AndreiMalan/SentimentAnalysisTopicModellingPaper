"""
Metrics and Evaluation Module
=============================
Comprehensive metrics for topic modeling and sentiment analysis:
- Topic distribution and coherence
- Keyword coverage and strength analysis
- Cross-algorithm agreement
- Brand-level comparative analysis
- Statistical significance tests
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from collections import Counter
from scipy.stats import entropy, chi2_contingency
from config.constructs import CONSTRUCTS, CONSTRUCT_NAMES


# ========================================================================
# TOPIC DISTRIBUTION METRICS
# ========================================================================

def topic_distribution(results_df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage per assigned topic."""
    counts = results_df["assigned_topic"].value_counts()
    pcts = results_df["assigned_topic"].value_counts(normalize=True) * 100
    return pd.DataFrame({"count": counts, "percentage": pcts.round(2)})


def topic_distribution_by_brand(results_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulation of topics by brand (counts)."""
    return pd.crosstab(results_df["brand"], results_df["assigned_topic"])


def topic_distribution_by_brand_pct(results_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulation of topics by brand (row-normalised %)."""
    ct = pd.crosstab(results_df["brand"], results_df["assigned_topic"])
    return ct.div(ct.sum(axis=1), axis=0).mul(100).round(2)


# ========================================================================
# CONFIDENCE / ACCURACY PROXIES
# ========================================================================

def confidence_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Mean, median, std confidence per topic."""
    return (
        results_df.groupby("assigned_topic")["confidence"]
        .agg(["mean", "median", "std", "count"])
        .round(4)
        .sort_values("mean", ascending=False)
    )


def confidence_by_brand(results_df: pd.DataFrame) -> pd.DataFrame:
    """Mean confidence per brand x topic."""
    return (
        results_df.groupby(["brand", "assigned_topic"])["confidence"]
        .mean()
        .unstack(fill_value=0)
        .round(4)
    )


# ========================================================================
# KEYWORD COVERAGE ANALYSIS
# ========================================================================

def keyword_hit_rate(texts: List[str], constructs: Dict = None) -> pd.DataFrame:
    """For each construct, compute what fraction of texts contain >= 1 keyword."""
    constructs = constructs or CONSTRUCTS
    rows = []
    for cname, cinfo in constructs.items():
        keywords = cinfo["inclusion_keywords"]
        hits = 0
        for text in texts:
            text_lower = text.lower()
            if any(kw.lower() in text_lower for kw in keywords):
                hits += 1
        rows.append({
            "construct": cname,
            "keyword_count": len(keywords),
            "documents_hit": hits,
            "hit_rate_pct": round(hits / max(len(texts), 1) * 100, 2),
        })
    return pd.DataFrame(rows)


def top_keywords_by_construct(
    texts: List[str], constructs: Dict = None, top_n: int = 10
) -> Dict[str, List[tuple]]:
    """For each construct, rank keywords by how often they appear in texts."""
    constructs = constructs or CONSTRUCTS
    result = {}
    for cname, cinfo in constructs.items():
        kw_counts = []
        for kw in cinfo["inclusion_keywords"]:
            count = sum(1 for t in texts if kw.lower() in t.lower())
            kw_counts.append((kw, count))
        kw_counts.sort(key=lambda x: x[1], reverse=True)
        result[cname] = kw_counts[:top_n]
    return result


# ========================================================================
# CROSS-ALGORITHM AGREEMENT
# ========================================================================

def pairwise_agreement(results_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute pairwise agreement rate between classification methods.

    Parameters
    ----------
    results_dict : dict
        Keys are method names, values are DataFrames with 'assigned_topic' column.
        All DataFrames must have the same length and ordering.
    """
    methods = [k for k in results_dict if not k.endswith("_model")]
    n = len(methods)
    matrix = pd.DataFrame(np.zeros((n, n)), index=methods, columns=methods)

    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i == j:
                matrix.iloc[i, j] = 1.0
            else:
                t1 = results_dict[m1]["assigned_topic"].values
                t2 = results_dict[m2]["assigned_topic"].values
                min_len = min(len(t1), len(t2))
                agreement = np.mean(t1[:min_len] == t2[:min_len])
                matrix.iloc[i, j] = round(agreement, 4)

    return matrix


def consensus_topics(results_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """For each document, compute the majority-vote topic across methods."""
    methods = [k for k in results_dict if not k.endswith("_model")]
    if not methods:
        return pd.DataFrame()

    n_docs = len(results_dict[methods[0]])
    rows = []
    for i in range(n_docs):
        votes = [results_dict[m]["assigned_topic"].iloc[i] for m in methods]
        counter = Counter(votes)
        top_topic, top_count = counter.most_common(1)[0]
        rows.append({
            "consensus_topic": top_topic,
            "agreement_ratio": top_count / len(methods),
            "n_methods_agree": top_count,
        })
    return pd.DataFrame(rows)


# ========================================================================
# BRAND COMPARATIVE ANALYSIS
# ========================================================================

def brand_topic_chi2(results_df: pd.DataFrame) -> Dict:
    """Chi-squared test for independence between brand and topic assignment."""
    ct = pd.crosstab(results_df["brand"], results_df["assigned_topic"])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return {"chi2": None, "p_value": None, "dof": None, "significant": None}
    chi2, p, dof, expected = chi2_contingency(ct)
    return {
        "chi2": round(chi2, 4),
        "p_value": round(p, 6),
        "dof": dof,
        "significant_at_005": p < 0.05,
        "expected_frequencies": pd.DataFrame(
            expected, index=ct.index, columns=ct.columns
        ).round(2),
    }


def brand_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Per-brand: dominant topic, avg confidence, comment count."""
    rows = []
    for brand in results_df["brand"].unique():
        bdf = results_df[results_df["brand"] == brand]
        dominant = bdf["assigned_topic"].value_counts().index[0]
        rows.append({
            "brand": brand,
            "n_comments": len(bdf),
            "dominant_topic": dominant,
            "dominant_topic_pct": round(
                bdf["assigned_topic"].value_counts(normalize=True).iloc[0] * 100, 2
            ),
            "avg_confidence": round(bdf["confidence"].mean(), 4),
            "unique_topics": bdf["assigned_topic"].nunique(),
        })
    return pd.DataFrame(rows)


# ========================================================================
# ENTROPY / DIVERSITY METRICS
# ========================================================================

def topic_entropy(results_df: pd.DataFrame) -> float:
    """Shannon entropy of the topic distribution (higher = more diverse)."""
    counts = results_df["assigned_topic"].value_counts(normalize=True).values
    return round(float(entropy(counts)), 4)


def brand_topic_entropy(results_df: pd.DataFrame) -> pd.DataFrame:
    """Entropy of topic distribution per brand."""
    rows = []
    for brand in results_df["brand"].unique():
        bdf = results_df[results_df["brand"] == brand]
        counts = bdf["assigned_topic"].value_counts(normalize=True).values
        rows.append({
            "brand": brand,
            "topic_entropy": round(float(entropy(counts)), 4),
            "n_topics": bdf["assigned_topic"].nunique(),
        })
    return pd.DataFrame(rows)


# ========================================================================
# COMPREHENSIVE REPORT
# ========================================================================

def generate_full_report(
    results_df: pd.DataFrame,
    texts: List[str],
    all_results: Dict[str, pd.DataFrame] = None,
) -> Dict:
    """Generate a comprehensive metrics report."""
    report = {
        "topic_distribution": topic_distribution(results_df),
        "confidence_summary": confidence_summary(results_df),
        "keyword_hit_rate": keyword_hit_rate(texts),
        "top_keywords": top_keywords_by_construct(texts),
        "brand_summary": brand_summary(results_df),
        "topic_entropy": topic_entropy(results_df),
        "brand_entropy": brand_topic_entropy(results_df),
        "chi2_test": brand_topic_chi2(results_df),
    }
    if results_df["brand"].notna().any():
        report["topic_by_brand"] = topic_distribution_by_brand(results_df)
        report["topic_by_brand_pct"] = topic_distribution_by_brand_pct(results_df)
        report["confidence_by_brand"] = confidence_by_brand(results_df)

    if all_results:
        report["pairwise_agreement"] = pairwise_agreement(all_results)
        report["consensus"] = consensus_topics(all_results)

    return report
