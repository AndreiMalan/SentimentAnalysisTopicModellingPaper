"""
Tab A: Topic Modeling
=====================
Run multiple topic-modeling algorithms, classify comments into the 7
literature-based constructs, and analyse results per company.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config.constructs import CONSTRUCTS, CONSTRUCT_NAMES
from pipeline.topic_modeling import (
    KeywordClassifier, LDATopicModel, NMFTopicModel, SeededLDA,
    BERTopicModel, ZeroShotClassifier,
    BERTOPIC_AVAILABLE, TRANSFORMERS_AVAILABLE,
    run_all_topic_models,
)


def render_tab():
    st.header("Topic Modeling")
    st.markdown(
        "Classify YouTube comments into **7 literature-based constructs** "
        "using multiple algorithms. Analyse topic distributions per brand."
    )

    if "cleaned_df" not in st.session_state:
        st.warning("Please run the cleaning pipeline first (Data Pipeline tab).")
        return

    cleaned_df = st.session_state["cleaned_df"]
    texts = cleaned_df["comment_processed"].tolist()
    brands = cleaned_df["Brand"].tolist()

    # --- Construct overview ---
    with st.expander("Literature Constructs (7 topics)"):
        for name, info in CONSTRUCTS.items():
            st.markdown(f"**{name}** - {info['definition'][:120]}...")
            st.caption(f"Keywords: {', '.join(info['inclusion_keywords'][:8])}...")

    # --- Method selection ---
    st.subheader("Select Methods")

    available_methods = ["Keyword", "LDA", "NMF", "Seeded LDA"]
    if BERTOPIC_AVAILABLE:
        available_methods.append("BERTopic")
    if TRANSFORMERS_AVAILABLE:
        available_methods.append("Zero-Shot")

    selected = st.multiselect(
        "Algorithms to run:",
        available_methods,
        default=["Keyword", "LDA", "NMF", "Seeded LDA"],
    )

    sample_size = st.slider(
        "Sample size (for speed)",
        min_value=100,
        max_value=min(len(texts), 10000),
        value=min(len(texts), 2000),
        step=100,
    )

    if st.button("Run Topic Modeling", type="primary"):
        # Sample
        idx = np.random.RandomState(42).choice(len(texts), size=sample_size, replace=False)
        sample_texts = [texts[i] for i in idx]
        sample_brands = [brands[i] for i in idx]

        progress = st.progress(0, text="Starting...")
        status = st.empty()

        def _cb(step, frac):
            progress.progress(min(frac, 1.0), text=f"{step}...")
            status.text(f"Running: {step}")

        with st.spinner("Running topic models..."):
            all_results = run_all_topic_models(
                sample_texts, sample_brands,
                methods=selected,
                progress_callback=_cb,
            )

        progress.progress(1.0, text="Done!")
        st.session_state["topic_results"] = all_results
        st.session_state["topic_texts"] = sample_texts
        st.session_state["topic_brands"] = sample_brands
        st.success(f"Completed {len([k for k in all_results if not k.endswith('_model')])} methods on {sample_size} comments.")

    # --- Display results ---
    if "topic_results" not in st.session_state:
        return

    all_results = st.session_state["topic_results"]
    method_names = [k for k in all_results if not k.endswith("_model")]

    # Pick primary method for detailed analysis
    primary = st.selectbox("Primary method for analysis:", method_names, index=0)
    primary_df = all_results[primary]
    st.session_state["primary_topic_df"] = primary_df

    # --- Overall topic distribution ---
    st.subheader(f"Topic Distribution ({primary})")
    fig, ax = plt.subplots(figsize=(10, 5))
    counts = primary_df["assigned_topic"].value_counts()
    colors = sns.color_palette("Set2", len(counts))
    counts.plot(kind="bar", ax=ax, color=colors)
    ax.set_ylabel("Count")
    ax.set_title(f"Comment Distribution by Construct ({primary})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Per-brand breakdown ---
    st.subheader("Topic Distribution by Brand")
    ct = pd.crosstab(primary_df["brand"], primary_df["assigned_topic"])
    ct_pct = ct.div(ct.sum(axis=1), axis=0).mul(100)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    ct.plot(kind="bar", stacked=True, ax=axes[0], colormap="Set2")
    axes[0].set_title("Counts")
    axes[0].set_ylabel("Comments")
    axes[0].legend(bbox_to_anchor=(1.05, 1), fontsize=7)
    plt.setp(axes[0].get_xticklabels(), rotation=0)

    ct_pct.plot(kind="bar", stacked=True, ax=axes[1], colormap="Set2")
    axes[1].set_title("Percentage")
    axes[1].set_ylabel("%")
    axes[1].legend(bbox_to_anchor=(1.05, 1), fontsize=7)
    plt.setp(axes[1].get_xticklabels(), rotation=0)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Heatmap: Brand x Topic ---
    st.subheader("Brand-Topic Heatmap (%)")
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(ct_pct, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
    ax.set_title("Topic distribution across brands (%)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Per-brand top topics ---
    st.subheader("Dominant Topics per Brand")
    for brand in primary_df["brand"].dropna().unique():
        bdf = primary_df[primary_df["brand"] == brand]
        top = bdf["assigned_topic"].value_counts().head(3)
        with st.expander(f"{brand} ({len(bdf)} comments)"):
            for topic, count in top.items():
                pct = count / len(bdf) * 100
                st.write(f"- **{topic}**: {count} ({pct:.1f}%)")

    # --- Confidence distribution ---
    st.subheader("Confidence Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    for topic in primary_df["assigned_topic"].unique():
        tdf = primary_df[primary_df["assigned_topic"] == topic]
        ax.hist(tdf["confidence"], alpha=0.5, label=topic, bins=20)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Confidence Distribution by Topic ({primary})")
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Multi-method comparison ---
    if len(method_names) > 1:
        st.subheader("Multi-Method Comparison")
        comparison_rows = []
        for m in method_names:
            mdf = all_results[m]
            comparison_rows.append({
                "Method": m,
                "Avg Confidence": round(mdf["confidence"].mean(), 4),
                "Unclassified %": round(
                    (mdf["assigned_topic"] == "Unclassified").mean() * 100, 2
                ),
                "Unique Topics": mdf["assigned_topic"].nunique(),
                "Dominant Topic": mdf["assigned_topic"].value_counts().index[0],
            })
        st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)

    # --- Topic words (for model-based methods) ---
    for m in method_names:
        model_key = f"{m}_model"
        if model_key in all_results:
            model = all_results[model_key]
            if hasattr(model, "topic_words") and model.topic_words:
                with st.expander(f"Discovered topic words ({m})"):
                    for tname, words in model.topic_words.items():
                        st.write(f"**{tname}**: {', '.join(words[:12])}")

    # --- Sample classifications ---
    st.subheader("Sample Classifications")
    display_cols = ["text", "brand", "assigned_topic", "confidence"]
    available = [c for c in display_cols if c in primary_df.columns]
    sample_display = primary_df[available].copy()
    sample_display["text"] = sample_display["text"].str[:120]
    st.dataframe(sample_display.head(30), use_container_width=True)

    # --- Download ---
    csv = primary_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download full results (CSV)",
        csv,
        f"topic_results_{primary}.csv",
        "text/csv",
    )
