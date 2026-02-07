"""
Tab B: Metrics & Comparative Analysis
======================================
Comprehensive metrics dashboard: accuracy proxies, topic coherence,
keyword strength, brand comparisons, algorithm agreement, trend analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config.constructs import CONSTRUCTS
from pipeline.metrics import (
    topic_distribution,
    topic_distribution_by_brand,
    topic_distribution_by_brand_pct,
    confidence_summary,
    confidence_by_brand,
    keyword_hit_rate,
    top_keywords_by_construct,
    pairwise_agreement,
    consensus_topics,
    brand_summary,
    brand_topic_chi2,
    topic_entropy,
    brand_topic_entropy,
    generate_full_report,
)


def render_tab():
    st.header("Metrics & Comparative Analysis")
    st.markdown(
        "Detailed evaluation metrics, keyword analysis, "
        "brand comparisons, and cross-algorithm agreement."
    )

    if "topic_results" not in st.session_state:
        st.warning("Run topic modeling first (Topic Modeling tab).")
        return

    all_results = st.session_state["topic_results"]
    texts = st.session_state.get("topic_texts", [])
    brands = st.session_state.get("topic_brands", [])
    method_names = [k for k in all_results if not k.endswith("_model")]

    primary_name = st.selectbox("Primary method:", method_names, index=0, key="metrics_primary")
    primary_df = all_results[primary_name]

    # =================================================================
    # SECTION 1: TOPIC DISTRIBUTION
    # =================================================================
    st.subheader("1. Topic Distribution")

    dist = topic_distribution(primary_df)
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(dist, use_container_width=True)
    with col2:
        ent = topic_entropy(primary_df)
        st.metric("Topic Entropy (diversity)", ent)
        unclassified_pct = (
            (primary_df["assigned_topic"] == "Unclassified").mean() * 100
        )
        st.metric("Unclassified %", f"{unclassified_pct:.1f}%")

    fig, ax = plt.subplots(figsize=(9, 5))
    dist["count"].plot(kind="barh", ax=ax, color=sns.color_palette("Set2", len(dist)))
    ax.set_xlabel("Count")
    ax.set_title(f"Topic Distribution ({primary_name})")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # =================================================================
    # SECTION 2: CONFIDENCE / ACCURACY PROXIES
    # =================================================================
    st.subheader("2. Classification Confidence")

    conf = confidence_summary(primary_df)
    st.dataframe(conf, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    conf["mean"].plot(kind="bar", ax=ax, yerr=conf["std"], color="steelblue", capsize=3)
    ax.set_ylabel("Mean Confidence")
    ax.set_title("Average Confidence per Topic")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Confidence by brand
    st.subheader("Confidence by Brand")
    conf_brand = confidence_by_brand(primary_df)
    if not conf_brand.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(conf_brand, annot=True, fmt=".3f", cmap="YlGn", ax=ax)
        ax.set_title("Mean Confidence per Brand x Topic")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # =================================================================
    # SECTION 3: KEYWORD ANALYSIS
    # =================================================================
    st.subheader("3. Keyword Coverage & Strength")

    hit_df = keyword_hit_rate(texts)
    st.dataframe(hit_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(hit_df["construct"], hit_df["hit_rate_pct"], color="teal")
    ax.set_xlabel("% of documents containing >= 1 keyword")
    ax.set_title("Keyword Hit Rate by Construct")
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Top keywords per construct
    st.subheader("Top Keywords by Construct")
    top_kw = top_keywords_by_construct(texts, top_n=8)
    for cname, kwlist in top_kw.items():
        with st.expander(f"{cname}"):
            kw_df = pd.DataFrame(kwlist, columns=["Keyword", "Occurrences"])
            st.dataframe(kw_df, use_container_width=True)

    # =================================================================
    # SECTION 4: BRAND COMPARATIVE ANALYSIS
    # =================================================================
    st.subheader("4. Brand Comparative Analysis")

    bs = brand_summary(primary_df)
    st.dataframe(bs, use_container_width=True)

    # Chi-squared test
    chi2 = brand_topic_chi2(primary_df)
    if chi2.get("chi2") is not None:
        st.markdown("**Chi-squared Test: Brand vs. Topic independence**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Chi2", chi2["chi2"])
        col2.metric("p-value", f"{chi2['p_value']:.6f}")
        col3.metric(
            "Significant (p < 0.05)",
            "Yes" if chi2["significant_at_005"] else "No",
        )

    # Brand topic distribution (%)
    pct_df = topic_distribution_by_brand_pct(primary_df)
    if not pct_df.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        pct_df.plot(kind="bar", ax=ax, colormap="Set2")
        ax.set_ylabel("%")
        ax.set_title("Topic Distribution by Brand (%)")
        ax.legend(bbox_to_anchor=(1.05, 1), fontsize=7)
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Brand entropy
    be = brand_topic_entropy(primary_df)
    if not be.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(be["brand"], be["topic_entropy"], color=sns.color_palette("Set2"))
        ax.set_ylabel("Shannon Entropy")
        ax.set_title("Topic Diversity per Brand (higher = more diverse)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # =================================================================
    # SECTION 5: CROSS-ALGORITHM AGREEMENT
    # =================================================================
    if len(method_names) > 1:
        st.subheader("5. Cross-Algorithm Agreement")

        agree_matrix = pairwise_agreement(all_results)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            agree_matrix.astype(float), annot=True, fmt=".3f",
            cmap="Blues", vmin=0, vmax=1, ax=ax,
        )
        ax.set_title("Pairwise Agreement Between Methods")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Consensus
        cons = consensus_topics(all_results)
        if not cons.empty:
            st.markdown("**Consensus (majority vote) statistics**")
            col1, col2 = st.columns(2)
            col1.metric(
                "Mean agreement ratio",
                f"{cons['agreement_ratio'].mean():.3f}",
            )
            col2.metric(
                "Full agreement %",
                f"{(cons['agreement_ratio'] == 1.0).mean() * 100:.1f}%",
            )

            cons_dist = cons["consensus_topic"].value_counts()
            fig, ax = plt.subplots(figsize=(10, 4))
            cons_dist.plot(kind="bar", ax=ax, color=sns.color_palette("Set2"))
            ax.set_title("Consensus Topic Distribution (majority vote)")
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # =================================================================
    # SECTION 6: DOWNLOAD FULL REPORT
    # =================================================================
    st.subheader("6. Export")
    report = generate_full_report(primary_df, texts, all_results)

    csv_parts = []
    for key, val in report.items():
        if isinstance(val, pd.DataFrame):
            csv_parts.append(f"--- {key} ---\n{val.to_csv()}\n")

    combined_csv = "\n".join(csv_parts).encode("utf-8")
    st.download_button(
        "Download Metrics Report (CSV)",
        combined_csv,
        "metrics_report.csv",
        "text/csv",
    )
