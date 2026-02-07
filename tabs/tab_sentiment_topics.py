"""
Tab C: Sentiment-Topic Integration
====================================
Combine the 6-emotion BERT sentiment analysis with topic classifications
to reveal which emotions dominate which constructs, per brand.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config.constructs import EMOTION_LABELS
from pipeline.sentiment_analysis import (
    run_sentiment_pipeline,
    TRANSFORMERS_AVAILABLE,
)
from pipeline.integration import (
    merge_topic_sentiment,
    topic_emotion_crosstab,
    topic_vader_summary,
    brand_topic_emotion,
    brand_emotion_summary,
    topic_sentiment_chi2,
    generate_insights,
    generate_integration_report,
)


def render_tab():
    st.header("Sentiment-Topic Integration")
    st.markdown(
        "Combine BERT emotion predictions (sadness, joy, love, anger, fear, surprise) "
        "with topic classifications to uncover **which emotions dominate which constructs** "
        "and how this varies across brands."
    )

    if "primary_topic_df" not in st.session_state:
        st.warning("Run topic modeling first (Topic Modeling tab) and select a primary method.")
        return

    topic_df = st.session_state["primary_topic_df"]
    texts = st.session_state.get("topic_texts", topic_df["text"].tolist())
    brands = st.session_state.get("topic_brands", topic_df["brand"].tolist() if "brand" in topic_df else None)

    st.info(f"Analysing {len(texts):,} comments from topic modeling results.")

    # --- Run sentiment ---
    use_bert = st.checkbox("Use BERT emotion model", value=TRANSFORMERS_AVAILABLE)

    if st.button("Run Sentiment Analysis & Integration", type="primary"):
        progress = st.progress(0, text="Starting sentiment analysis...")
        status = st.empty()

        def _cb(step, frac):
            progress.progress(min(frac, 1.0), text=f"{step}...")
            status.text(f"Step: {step}")

        with st.spinner("Running sentiment analysis..."):
            sent_df = run_sentiment_pipeline(
                texts, use_bert=use_bert, progress_callback=_cb,
            )

        progress.progress(1.0, text="Done!")

        merged = merge_topic_sentiment(topic_df, sent_df, texts, brands)
        st.session_state["merged_df"] = merged
        st.success("Sentiment analysis complete. Results integrated.")

    if "merged_df" not in st.session_state:
        return

    merged = st.session_state["merged_df"]

    # =================================================================
    # SECTION 1: OVERALL EMOTION DISTRIBUTION
    # =================================================================
    st.subheader("1. Overall Emotion Distribution")

    if "emotion" in merged.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Emotion counts
        emotion_counts = merged["emotion"].value_counts()
        colors = sns.color_palette("husl", len(emotion_counts))
        emotion_counts.plot(kind="bar", ax=axes[0], color=colors)
        axes[0].set_title("Emotion Counts (BERT)")
        axes[0].set_ylabel("Count")
        plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right")

        # VADER distribution
        if "vader_compound" in merged.columns:
            axes[1].hist(merged["vader_compound"], bins=40, color="steelblue", alpha=0.7)
            axes[1].axvline(0, color="red", linestyle="--", alpha=0.5)
            axes[1].set_title("VADER Compound Score Distribution")
            axes[1].set_xlabel("Compound Score")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # =================================================================
    # SECTION 2: TOPIC-EMOTION HEATMAP
    # =================================================================
    st.subheader("2. Topic-Emotion Heatmap")

    te = topic_emotion_crosstab(merged)
    if te and "percentages" in te:
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        sns.heatmap(te["counts"], annot=True, fmt="d", cmap="YlOrRd", ax=axes[0])
        axes[0].set_title("Topic x Emotion (counts)")
        axes[0].set_xlabel("Emotion")
        axes[0].set_ylabel("Topic")

        sns.heatmap(te["percentages"], annot=True, fmt=".1f", cmap="YlOrRd", ax=axes[1])
        axes[1].set_title("Topic x Emotion (% within topic)")
        axes[1].set_xlabel("Emotion")
        axes[1].set_ylabel("Topic")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Dominant emotion per topic
        st.markdown("**Dominant emotion per topic:**")
        for topic, info in te["dominant_emotion_per_topic"].items():
            st.write(f"- **{topic}**: {info['emotion']} ({info['pct']:.1f}%)")

    # =================================================================
    # SECTION 3: VADER BY TOPIC
    # =================================================================
    if "vader_compound" in merged.columns:
        st.subheader("3. Sentiment Intensity by Topic (VADER)")

        vader_summary = topic_vader_summary(merged)
        st.dataframe(vader_summary, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        vader_summary["mean"].sort_values().plot(
            kind="barh", ax=ax, color="teal", xerr=vader_summary["std"]
        )
        ax.axvline(0, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Mean VADER Compound")
        ax.set_title("Average Sentiment per Topic")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # =================================================================
    # SECTION 4: BRAND-TOPIC-EMOTION ANALYSIS
    # =================================================================
    st.subheader("4. Brand-Topic-Emotion Breakdown")

    bte = brand_topic_emotion(merged)
    if not bte.empty:
        st.dataframe(bte, use_container_width=True)

        # Pivot for heatmap: brand vs dominant_emotion coloured by count
        for brand in merged["brand"].dropna().unique():
            brand_data = merged[merged["brand"] == brand]
            with st.expander(f"{brand} - Emotion by Topic"):
                if "emotion" in brand_data.columns:
                    ct = pd.crosstab(
                        brand_data["assigned_topic"], brand_data["emotion"]
                    )
                    ct_pct = ct.div(ct.sum(axis=1), axis=0).mul(100).round(1)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.heatmap(ct_pct, annot=True, fmt=".1f", cmap="coolwarm", ax=ax)
                    ax.set_title(f"{brand}: Emotion distribution within each topic (%)")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

    # Brand emotion summary
    st.subheader("5. Overall Emotion Distribution per Brand")
    be = brand_emotion_summary(merged)
    if not be.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        be.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
        ax.set_ylabel("% of comments")
        ax.set_title("Emotion Distribution Across Brands")
        ax.legend(bbox_to_anchor=(1.05, 1), fontsize=8)
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # =================================================================
    # SECTION 5: STATISTICAL TESTS
    # =================================================================
    st.subheader("6. Statistical Association")

    chi2_res = topic_sentiment_chi2(merged)
    if chi2_res.get("chi2") is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Chi2 (topic x emotion)", chi2_res["chi2"])
        col2.metric("p-value", f"{chi2_res['p_value']:.6f}")
        col3.metric(
            "Significant (p < 0.05)",
            "Yes" if chi2_res["significant_at_005"] else "No",
        )

    # =================================================================
    # SECTION 6: KEY INSIGHTS
    # =================================================================
    st.subheader("7. Key Insights")
    insights = generate_insights(merged)
    for insight in insights:
        st.markdown(f"- {insight}")

    # =================================================================
    # DOWNLOAD
    # =================================================================
    st.subheader("8. Export")
    csv = merged.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download integrated results (CSV)",
        csv,
        "sentiment_topic_integration.csv",
        "text/csv",
    )
