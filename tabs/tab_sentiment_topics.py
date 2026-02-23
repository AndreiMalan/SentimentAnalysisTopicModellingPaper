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

    if "topic_results" not in st.session_state:
        st.warning("Run topic modeling first (Topic Modeling tab).")
        return

    all_results = st.session_state["topic_results"]
    method_names = [k for k in all_results if not k.endswith("_model")]

    if not method_names:
        st.warning("No topic modeling results available. Run topic modeling first.")
        return

    # --- Method selector ---
    st.subheader("Select Topic Method for Sentiment Integration")
    selected_method = st.selectbox(
        "Topic method to integrate with sentiment:",
        method_names,
        index=0,
        key="sentiment_method_selector",
    )
    topic_df = all_results[selected_method]
    texts = st.session_state.get("topic_texts", topic_df["text"].tolist())
    brands = st.session_state.get("topic_brands", topic_df["brand"].tolist() if "brand" in topic_df else None)

    st.info(f"Analysing **{len(texts):,}** comments using **{selected_method}** topic assignments.")

    # --- Topic composition (what each topic consists of) ---
    model_key = f"{selected_method}_model"
    if model_key in all_results:
        _model = all_results[model_key]
        if hasattr(_model, "topic_words") and _model.topic_words:
            with st.expander(f"Topic Composition — {selected_method} (top words per topic)", expanded=True):
                n_topics = len(_model.topic_words)
                n_cols = min(3, n_topics)
                n_rows = (n_topics + n_cols - 1) // n_cols
                fig_tw, axes_tw = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.5 * n_rows))
                if n_topics == 1:
                    axes_tw = np.array([axes_tw])
                axes_tw = np.atleast_2d(axes_tw)
                colors_tw = sns.color_palette("Set2", n_topics)

                for i, (tname, ww) in enumerate(_model.topic_words.items()):
                    r, c = divmod(i, n_cols)
                    ax = axes_tw[r, c]
                    top20 = ww[:20]
                    words = [w for w, _ in top20][::-1]
                    weights = [v for _, v in top20][::-1]
                    ax.barh(words, weights, color=colors_tw[i % len(colors_tw)])
                    ax.set_title(tname, fontsize=10, fontweight="bold")
                    ax.set_xlabel("Weight")

                for j in range(n_topics, n_rows * n_cols):
                    r, c = divmod(j, n_cols)
                    axes_tw[r, c].set_visible(False)

                plt.suptitle(f"{selected_method} — Top Words per Topic", fontsize=13, y=1.01)
                plt.tight_layout()
                st.pyplot(fig_tw)
                plt.close()

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
    # SECTION 1: OVERALL EMOTION DISTRIBUTION — TOP 3 HIGHLIGHTED
    # =================================================================
    st.subheader("1. Overall Emotion Distribution")

    if "emotion" in merged.columns:
        emotion_counts = merged["emotion"].value_counts()
        total = len(merged)

        # --- Top 3 Emotions as prominent metric cards ---
        top_n = min(3, len(emotion_counts))
        cols_top = st.columns(top_n)
        medal = ["1st", "2nd", "3rd"]
        for rank in range(top_n):
            emo = emotion_counts.index[rank]
            cnt = emotion_counts.iloc[rank]
            pct = cnt / total * 100
            cols_top[rank].metric(
                f"{medal[rank]} — {emo.capitalize()}",
                f"{pct:.1f}%",
                f"{cnt:,} comments",
            )

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

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

        # Top 3 emotions per topic
        st.markdown("**Top 3 emotions per topic:**")
        for topic in te["percentages"].index:
            row = te["percentages"].loc[topic].sort_values(ascending=False)
            top3 = row.head(3)
            parts = [f"{emo} ({pct:.1f}%)" for emo, pct in top3.items()]
            st.write(f"- **{topic}**: {', '.join(parts)}")

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
