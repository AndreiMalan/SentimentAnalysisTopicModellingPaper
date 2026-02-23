"""
Tab A: Topic Modeling
=====================
Run LDA (unsupervised), Seeded LDA, and BERTopic.
LDA includes k-selection with perplexity/coherence curves and
LDAvis-style per-topic word charts.  All metrics are printed to
the console for article writing.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config.constructs import CONSTRUCTS, CONSTRUCT_NAMES
from pipeline.topic_modeling import (
    LDATopicModel, SeededLDA, BERTopicModel,
    BERTOPIC_AVAILABLE,
    run_all_topic_models,
)


def _show_ngram_table(topic_words, method_label):
    """Display a table of bigrams and trigrams per topic, extracted from topic_words."""
    rows = []
    for tname, ww in topic_words.items():
        bigrams = [(w, v) for w, v in ww if w.count(" ") == 1]
        trigrams = [(w, v) for w, v in ww if w.count(" ") == 2]
        top_bi = bigrams[:10]
        top_tri = trigrams[:10]
        n_show = max(len(top_bi), len(top_tri), 1)
        for rank in range(n_show):
            bi_str = f"{top_bi[rank][0]} ({top_bi[rank][1]:.4f})" if rank < len(top_bi) else ""
            tri_str = f"{top_tri[rank][0]} ({top_tri[rank][1]:.4f})" if rank < len(top_tri) else ""
            rows.append({
                "Topic": tname if rank == 0 else "",
                "Bigram": bi_str,
                "Trigram": tri_str,
            })
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info(f"No bigrams/trigrams found for {method_label}.")


def render_tab():
    st.header("Topic Modeling")
    st.markdown(
        "Three algorithms: **LDA** (unsupervised, discovers topics), "
        "**Seeded LDA** (guided by construct keywords), and "
        "**BERTopic** (transformer-based, guided).  "
        "LDA tests k = 4 … 15 and selects the optimal number of topics "
        "via coherence and perplexity."
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
    st.subheader("Configuration")

    available_methods = ["LDA", "Seeded LDA"]
    if BERTOPIC_AVAILABLE:
        available_methods.append("BERTopic")

    selected = st.multiselect(
        "Algorithms to run:",
        available_methods,
        default=["LDA", "Seeded LDA"],
    )

    col_cfg1, col_cfg2 = st.columns(2)
    sample_size = col_cfg1.slider(
        "Sample size",
        min_value=100,
        max_value=min(len(texts), 10000),
        value=min(len(texts), 2000),
        step=100,
    )
    lda_k_min, lda_k_max = col_cfg2.slider(
        "LDA k range",
        min_value=6, max_value=20, value=(6, 15),
        key="lda_k_range_slider",
    )

    if st.button("Run Topic Modeling", type="primary"):
        # Read slider values BEFORE any rerun
        _k_min = int(lda_k_min)
        _k_max = int(lda_k_max)
        _sample = int(sample_size)

        idx = np.random.RandomState(42).choice(len(texts), size=min(_sample, len(texts)), replace=False)
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
                lda_k_min=_k_min,
                lda_k_max=_k_max,
                progress_callback=_cb,
            )

        progress.progress(1.0, text="Done!")
        st.session_state["topic_results"] = all_results
        st.session_state["topic_texts"] = sample_texts
        st.session_state["topic_brands"] = sample_brands
        st.success(f"Completed {len([k for k in all_results if not k.endswith('_model')])} methods on {min(sample_size, len(texts))} comments.")

    # --- Display results ---
    if "topic_results" not in st.session_state:
        return

    all_results = st.session_state["topic_results"]
    method_names = [k for k in all_results if not k.endswith("_model")]

    # ==================================================================
    # SECTION 1 — LDA  (unsupervised)
    # ==================================================================
    if "LDA" in all_results and "LDA_model" in all_results:
        lda_model: LDATopicModel = all_results["LDA_model"]
        lda_df = all_results["LDA"]

        st.markdown("---")
        st.subheader("1. LDA — Unsupervised Topic Discovery")

        # ----- 1a. K-Selection Metrics Table -----
        st.markdown("#### 1a. Optimal k Selection (k = {} … {})".format(lda_model.k_min, lda_model.k_max))

        k_df = lda_model.get_k_metrics_df()
        st.dataframe(k_df, use_container_width=True)

        # Summary metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Optimal k", lda_model.optimal_k)
        col_m2.metric("Best Coherence (UMass)", f"{lda_model.coherence_umass[lda_model.optimal_k]:.4f}")
        col_m3.metric("Perplexity @ optimal k", f"{lda_model.perplexity_scores[lda_model.optimal_k]:.2f}")
        col_m4.metric("Topic Diversity", f"{lda_model.overall_diversity:.4f}")

        # ----- 1b. Coherence + Perplexity Plots -----
        st.markdown("#### 1b. Coherence & Perplexity vs. k")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ks = sorted(lda_model.perplexity_scores.keys())

        # Coherence (UMass)
        umass_vals = [lda_model.coherence_umass[k] for k in ks]
        axes[0].plot(ks, umass_vals, "o-", color="steelblue", linewidth=2)
        axes[0].axvline(lda_model.optimal_k, color="red", linestyle="--", alpha=0.7, label=f"optimal k={lda_model.optimal_k}")
        axes[0].set_xlabel("Number of Topics (k)")
        axes[0].set_ylabel("Coherence (UMass)")
        axes[0].set_title("UMass Coherence vs. k\n(closer to 0 = better)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Perplexity
        perp_vals = [lda_model.perplexity_scores[k] for k in ks]
        axes[1].plot(ks, perp_vals, "o-", color="darkorange", linewidth=2)
        axes[1].axvline(lda_model.optimal_k, color="red", linestyle="--", alpha=0.7, label=f"optimal k={lda_model.optimal_k}")
        axes[1].set_xlabel("Number of Topics (k)")
        axes[1].set_ylabel("Perplexity")
        axes[1].set_title("Perplexity vs. k\n(lower = better)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # NPMI
        npmi_vals = [lda_model.coherence_npmi[k] for k in ks]
        axes[2].plot(ks, npmi_vals, "o-", color="seagreen", linewidth=2)
        axes[2].axvline(lda_model.optimal_k, color="red", linestyle="--", alpha=0.7, label=f"optimal k={lda_model.optimal_k}")
        axes[2].set_xlabel("Number of Topics (k)")
        axes[2].set_ylabel("Coherence (NPMI)")
        axes[2].set_title("NPMI Coherence vs. k\n(higher = better)")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ----- 1c. LDAvis-style Topic Words -----
        st.markdown(f"#### 1c. Discovered Topics — Top Words (k = {lda_model.optimal_k})")

        # Topic proportions bar
        fig_prop, ax_prop = plt.subplots(figsize=(10, 4))
        topic_names = list(lda_model.topic_words.keys())
        proportions = lda_model.topic_proportions
        colors = sns.color_palette("Set2", len(topic_names))
        ax_prop.bar(topic_names, proportions, color=colors)
        ax_prop.set_ylabel("Mean Document Proportion")
        ax_prop.set_title("Topic Proportions (LDA)")
        for i, (name, prop) in enumerate(zip(topic_names, proportions)):
            ax_prop.text(i, prop + 0.005, f"{prop:.3f}", ha="center", fontsize=9)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig_prop)
        plt.close()

        # Per-topic word bar charts (LDAvis core)
        n_topics = len(lda_model.topic_words)
        n_cols = min(3, n_topics)
        n_rows = (n_topics + n_cols - 1) // n_cols
        fig_words, axes_w = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        if n_topics == 1:
            axes_w = np.array([axes_w])
        axes_w = np.atleast_2d(axes_w)

        for i, (tname, ww) in enumerate(lda_model.topic_words.items()):
            r, c = divmod(i, n_cols)
            ax = axes_w[r, c]
            top15 = ww[:20]
            words = [w for w, _ in top15][::-1]
            weights = [v for _, v in top15][::-1]
            ax.barh(words, weights, color=colors[i % len(colors)])
            ax.set_title(f"{tname} ({proportions[i]:.3f})", fontsize=10)
            ax.set_xlabel("Probability")

        # Hide unused axes
        for j in range(n_topics, n_rows * n_cols):
            r, c = divmod(j, n_cols)
            axes_w[r, c].set_visible(False)

        plt.suptitle("LDAvis — Top Words per Discovered Topic", fontsize=13, y=1.01)
        plt.tight_layout()
        st.pyplot(fig_words)
        plt.close()

        # ----- 1c-bis. Bigrams & Trigrams per Topic -----
        st.markdown(f"#### 1c'. Bigrams & Trigrams per Discovered Topic (k = {lda_model.optimal_k})")
        _show_ngram_table(lda_model.topic_words, "LDA")

        # ----- 1d. Topic Distribution -----
        st.markdown("#### 1d. Document Assignment")
        fig_dist, ax_dist = plt.subplots(figsize=(10, 5))
        counts = lda_df["assigned_topic"].value_counts().sort_index()
        counts.plot(kind="bar", ax=ax_dist, color=colors[:len(counts)])
        ax_dist.set_ylabel("Count")
        ax_dist.set_title("Document Distribution by LDA Topic")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig_dist)
        plt.close()

        # Per-brand
        if lda_df["brand"].notna().any():
            ct = pd.crosstab(lda_df["brand"], lda_df["assigned_topic"])
            ct_pct = ct.div(ct.sum(axis=1), axis=0).mul(100)
            fig_brand, axes_b = plt.subplots(1, 2, figsize=(16, 5))
            ct.plot(kind="bar", stacked=True, ax=axes_b[0], colormap="Set2")
            axes_b[0].set_title("LDA Topics by Brand (counts)")
            axes_b[0].legend(bbox_to_anchor=(1.05, 1), fontsize=7)
            plt.setp(axes_b[0].get_xticklabels(), rotation=0)

            ct_pct.plot(kind="bar", stacked=True, ax=axes_b[1], colormap="Set2")
            axes_b[1].set_title("LDA Topics by Brand (%)")
            axes_b[1].legend(bbox_to_anchor=(1.05, 1), fontsize=7)
            plt.setp(axes_b[1].get_xticklabels(), rotation=0)

            plt.tight_layout()
            st.pyplot(fig_brand)
            plt.close()

    # ==================================================================
    # SECTION 2 — SEEDED LDA  (semi-supervised)
    # ==================================================================
    if "Seeded LDA" in all_results and "Seeded LDA_model" in all_results:
        slda_model: SeededLDA = all_results["Seeded LDA_model"]
        slda_df = all_results["Seeded LDA"]

        st.markdown("---")
        st.subheader("2. Seeded LDA — Construct-Guided")

        # Summary metrics
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Perplexity", f"{slda_model.perplexity:.2f}")
        col_s2.metric("Coherence (UMass)", f"{slda_model.coherence_umass_val:.4f}")
        col_s3.metric("Coherence (NPMI)", f"{slda_model.coherence_npmi_val:.4f}")
        col_s4.metric("Topic Diversity", f"{slda_model.topic_diversity_val:.4f}")

        # Topic distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        counts = slda_df["assigned_topic"].value_counts()
        colors_s = sns.color_palette("Set2", len(counts))
        counts.plot(kind="bar", ax=ax, color=colors_s)
        ax.set_ylabel("Count")
        ax.set_title("Seeded LDA — Document Distribution by Construct")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Per-brand breakdown
        if slda_df["brand"].notna().any():
            st.markdown("**Brand-Construct Heatmap (%)**")
            ct = pd.crosstab(slda_df["brand"], slda_df["assigned_topic"])
            ct_pct = ct.div(ct.sum(axis=1), axis=0).mul(100)
            fig, ax = plt.subplots(figsize=(12, 4))
            sns.heatmap(ct_pct, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
            ax.set_title("Seeded LDA — Topic distribution across brands (%)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Topic words
        with st.expander("Seeded LDA — Top words per construct"):
            for tname, ww in slda_model.topic_words.items():
                words_str = ", ".join([f"{w} ({v:.4f})" for w, v in ww[:20]])
                st.markdown(f"**{tname}**: {words_str}")

        # Bigrams & Trigrams
        st.markdown("#### Seeded LDA — Bigrams & Trigrams per Construct")
        _show_ngram_table(slda_model.topic_words, "Seeded LDA")

    # ==================================================================
    # SECTION 3 — BERTopic  (transformer-based)
    # ==================================================================
    if "BERTopic" in all_results and "BERTopic_model" in all_results:
        bt_model: BERTopicModel = all_results["BERTopic_model"]
        bt_df = all_results["BERTopic"]

        st.markdown("---")
        st.subheader("3. BERTopic — Transformer-Based")

        fig, ax = plt.subplots(figsize=(10, 5))
        counts = bt_df["assigned_topic"].value_counts()
        counts.plot(kind="bar", ax=ax, color=sns.color_palette("Set2", len(counts)))
        ax.set_ylabel("Count")
        ax.set_title("BERTopic — Document Distribution by Construct")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        if bt_df["brand"].notna().any():
            st.markdown("**Brand-Construct Heatmap (%)**")
            ct = pd.crosstab(bt_df["brand"], bt_df["assigned_topic"])
            ct_pct = ct.div(ct.sum(axis=1), axis=0).mul(100)
            fig, ax = plt.subplots(figsize=(12, 4))
            sns.heatmap(ct_pct, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
            ax.set_title("BERTopic — Topic distribution across brands (%)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with st.expander("BERTopic — Topic words"):
            for tname, ww in bt_model.topic_words.items():
                construct = bt_model.topic_mapping.get(int(tname.split("_")[-1]), "?")
                words_str = ", ".join([f"{w} ({v:.4f})" for w, v in ww[:20]])
                st.markdown(f"**{tname}** -> {construct}: {words_str}")

        # Bigrams & Trigrams
        st.markdown("#### BERTopic — Bigrams & Trigrams per Topic")
        _show_ngram_table(bt_model.topic_words, "BERTopic")

    # ==================================================================
    # SECTION 4 — MULTI-METHOD COMPARISON
    # ==================================================================
    if len(method_names) > 1:
        st.markdown("---")
        st.subheader("4. Multi-Method Comparison")

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

    # ==================================================================
    # SECTION 5 — PRIMARY METHOD SELECTION & DOWNLOAD
    # ==================================================================
    st.markdown("---")
    st.subheader("5. Select Primary Method for Downstream Analysis")

    primary = st.selectbox("Primary method:", method_names, index=0)
    primary_df = all_results[primary]
    st.session_state["primary_topic_df"] = primary_df

    # Confidence distribution
    fig, ax = plt.subplots(figsize=(10, 4))
    for topic in primary_df["assigned_topic"].unique():
        tdf = primary_df[primary_df["assigned_topic"] == topic]
        ax.hist(tdf["confidence"], alpha=0.5, label=topic, bins=20)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Confidence Distribution ({primary})")
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Sample classifications
    st.markdown("**Sample Classifications**")
    display_cols = ["text", "brand", "assigned_topic", "confidence"]
    available = [c for c in display_cols if c in primary_df.columns]
    sample_display = primary_df[available].copy()
    sample_display["text"] = sample_display["text"].str[:120]
    st.dataframe(sample_display.head(30), use_container_width=True)

    # Download
    csv = primary_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download full results (CSV)",
        csv,
        f"topic_results_{primary}.csv",
        "text/csv",
    )
