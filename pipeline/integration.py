"""
Topic-Sentiment Integration Module
===================================
Combine topic classifications with sentiment predictions to produce
joint analyses: which emotions dominate which constructs, how sentiment
varies across brands and topics, and statistical summaries.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from scipy.stats import chi2_contingency
from config.constructs import EMOTION_LABELS


# ========================================================================
# CORE INTEGRATION
# ========================================================================

def merge_topic_sentiment(
    topic_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    texts: List[str] = None,
    brands: List[str] = None,
) -> pd.DataFrame:
    """Merge topic predictions with sentiment predictions into a single DataFrame.

    Both inputs must have the same number of rows (aligned by index).
    """
    merged = topic_df[["assigned_topic", "confidence", "method"]].copy()
    merged = merged.rename(columns={"confidence": "topic_confidence"})

    # Sentiment columns
    for col in sentiment_df.columns:
        merged[col] = sentiment_df[col].values

    if texts is not None:
        merged.insert(0, "text", texts)
    if brands is not None:
        merged.insert(1, "brand", brands)

    return merged


# ========================================================================
# TOPIC-SENTIMENT CROSS-TABULATION
# ========================================================================

def topic_emotion_crosstab(merged_df: pd.DataFrame) -> Dict:
    """Cross-tabulate assigned_topic vs. emotion (BERT)."""
    if "emotion" not in merged_df.columns:
        return {}

    ct = pd.crosstab(merged_df["assigned_topic"], merged_df["emotion"])
    ct_pct = ct.div(ct.sum(axis=1), axis=0).mul(100).round(2)

    dominant = {}
    for topic in ct.index:
        dom_emotion = ct.loc[topic].idxmax()
        dom_pct = ct_pct.loc[topic, dom_emotion]
        dominant[topic] = {"emotion": dom_emotion, "pct": dom_pct}

    return {
        "counts": ct,
        "percentages": ct_pct,
        "dominant_emotion_per_topic": dominant,
    }


def topic_vader_summary(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Mean VADER compound score per topic."""
    if "vader_compound" not in merged_df.columns:
        return pd.DataFrame()
    return (
        merged_df.groupby("assigned_topic")["vader_compound"]
        .agg(["mean", "median", "std", "count"])
        .round(4)
        .sort_values("mean", ascending=False)
    )


# ========================================================================
# BRAND-TOPIC-SENTIMENT ANALYSIS
# ========================================================================

def brand_topic_emotion(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Dominant emotion per brand-topic combination."""
    if "emotion" not in merged_df.columns or "brand" not in merged_df.columns:
        return pd.DataFrame()

    rows = []
    for brand in merged_df["brand"].dropna().unique():
        bdf = merged_df[merged_df["brand"] == brand]
        for topic in bdf["assigned_topic"].unique():
            tdf = bdf[bdf["assigned_topic"] == topic]
            if len(tdf) == 0:
                continue
            dom_emotion = tdf["emotion"].value_counts().index[0]
            dom_pct = tdf["emotion"].value_counts(normalize=True).iloc[0] * 100
            rows.append({
                "brand": brand,
                "topic": topic,
                "n_comments": len(tdf),
                "dominant_emotion": dom_emotion,
                "dominant_emotion_pct": round(dom_pct, 2),
                "avg_vader": round(tdf["vader_compound"].mean(), 4) if "vader_compound" in tdf else None,
            })
    return pd.DataFrame(rows)


def brand_emotion_summary(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Overall emotion distribution per brand."""
    if "emotion" not in merged_df.columns or "brand" not in merged_df.columns:
        return pd.DataFrame()

    ct = pd.crosstab(merged_df["brand"], merged_df["emotion"])
    ct_pct = ct.div(ct.sum(axis=1), axis=0).mul(100).round(2)
    return ct_pct


def topic_sentiment_chi2(merged_df: pd.DataFrame) -> Dict:
    """Chi-squared test: is there a significant association between topic and emotion?"""
    if "emotion" not in merged_df.columns:
        return {}
    ct = pd.crosstab(merged_df["assigned_topic"], merged_df["emotion"])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return {"chi2": None, "p_value": None}
    chi2, p, dof, _ = chi2_contingency(ct)
    return {
        "chi2": round(chi2, 4),
        "p_value": round(p, 6),
        "dof": dof,
        "significant_at_005": p < 0.05,
    }


# ========================================================================
# INSIGHTS GENERATION
# ========================================================================

def generate_insights(merged_df: pd.DataFrame) -> List[str]:
    """Generate human-readable insights from the integrated analysis."""
    insights = []

    # 1. Top 3 emotions
    if "emotion" in merged_df.columns:
        dom = merged_df["emotion"].value_counts()
        top_n = min(3, len(dom))
        parts = []
        for rank in range(top_n):
            emo = dom.index[rank]
            cnt = dom.iloc[rank]
            pct = cnt / len(merged_df) * 100
            parts.append(f"**{emo}** ({cnt:,} comments, {pct:.1f}%)")
        insights.append(
            f"Top {top_n} emotions across all comments: " + ", ".join(parts) + "."
        )

    # 2. Topic with most positive sentiment
    if "vader_compound" in merged_df.columns:
        topic_vader = merged_df.groupby("assigned_topic")["vader_compound"].mean()
        most_pos = topic_vader.idxmax()
        most_neg = topic_vader.idxmin()
        insights.append(
            f"**{most_pos}** has the most positive sentiment "
            f"(avg VADER: {topic_vader[most_pos]:.3f}), while "
            f"**{most_neg}** has the most negative "
            f"(avg VADER: {topic_vader[most_neg]:.3f})."
        )

    # 3. Brand differences
    if "brand" in merged_df.columns and merged_df["brand"].notna().any():
        brand_vader = merged_df.groupby("brand")["vader_compound"].mean()
        best_brand = brand_vader.idxmax()
        insights.append(
            f"Among brands, **{best_brand}** receives the most positive "
            f"overall sentiment (avg VADER: {brand_vader[best_brand]:.3f})."
        )

    # 4. Topic-emotion associations
    te = topic_emotion_crosstab(merged_df)
    if "dominant_emotion_per_topic" in te:
        for topic, info in te["dominant_emotion_per_topic"].items():
            if info["pct"] > 35:
                insights.append(
                    f"**{topic}** is strongly associated with "
                    f"**{info['emotion']}** ({info['pct']:.1f}% of comments)."
                )

    # 5. Chi-squared result
    chi2_res = topic_sentiment_chi2(merged_df)
    if chi2_res.get("p_value") is not None:
        if chi2_res["significant_at_005"]:
            insights.append(
                f"Chi-squared test confirms a statistically significant "
                f"association between topics and emotions "
                f"(chi2={chi2_res['chi2']}, p={chi2_res['p_value']:.6f})."
            )

    return insights


# ========================================================================
# FULL INTEGRATION REPORT
# ========================================================================

def generate_integration_report(merged_df: pd.DataFrame) -> Dict:
    """Generate full integration report."""
    return {
        "topic_emotion_crosstab": topic_emotion_crosstab(merged_df),
        "topic_vader_summary": topic_vader_summary(merged_df),
        "brand_topic_emotion": brand_topic_emotion(merged_df),
        "brand_emotion_summary": brand_emotion_summary(merged_df),
        "chi2_test": topic_sentiment_chi2(merged_df),
        "insights": generate_insights(merged_df),
    }
