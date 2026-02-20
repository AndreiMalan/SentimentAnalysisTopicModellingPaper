"""
Tab B: Metrics & Comparative Analysis
======================================
Research-grade metrics dashboard: LDA model selection, topic coherence,
confidence, keyword strength, brand comparisons, algorithm agreement.
All numbers are also printed to the console for article writing.
"""

import sys
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


def _print_section(title):
    """Print a section header to the console."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    sys.stdout.flush()


def render_tab():
    st.header("Metrics & Comparative Analysis")
    st.markdown(
        "Research-grade evaluation metrics for topic modeling results. "
        "All numerical results are also **printed to the console** "
        "so you can copy exact values for your article."
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

    # ==================================================================
    # SECTION 1: LDA MODEL SELECTION METRICS
    # ==================================================================
    if "LDA_model" in all_results:
        lda_model = all_results["LDA_model"]
        st.subheader("1. LDA Model Selection Metrics")

        k_df = lda_model.get_k_metrics_df()
        st.dataframe(k_df, use_container_width=True)

        # Print to console
        _print_section("LDA MODEL SELECTION METRICS")
        print(k_df.to_string(index=False))
        print(f"\nOptimal k: {lda_model.optimal_k}")
        print(f"Best UMass Coherence: {lda_model.coherence_umass[lda_model.optimal_k]:.4f}")
        print(f"Best NPMI Coherence: {lda_model.coherence_npmi[lda_model.optimal_k]:.4f}")
        print(f"Perplexity at optimal k: {lda_model.perplexity_scores[lda_model.optimal_k]:.2f}")
        print(f"Log-Likelihood at optimal k: {lda_model.log_likelihood[lda_model.optimal_k]:.2f}")
        print(f"Topic Diversity: {lda_model.overall_diversity:.4f}")

        # Dual-axis plot: coherence + perplexity
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ks = sorted(lda_model.perplexity_scores.keys())
        umass = [lda_model.coherence_umass[k] for k in ks]
        perp = [lda_model.perplexity_scores[k] for k in ks]

        color1 = "steelblue"
        ax1.set_xlabel("Number of Topics (k)", fontsize=11)
        ax1.set_ylabel("Coherence (UMass)", color=color1, fontsize=11)
        ax1.plot(ks, umass, "o-", color=color1, linewidth=2, label="Coherence (UMass)")
        ax1.tick_params(axis="y", labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = "darkorange"
        ax2.set_ylabel("Perplexity", color=color2, fontsize=11)
        ax2.plot(ks, perp, "s--", color=color2, linewidth=2, label="Perplexity")
        ax2.tick_params(axis="y", labelcolor=color2)

        ax1.axvline(lda_model.optimal_k, color="red", linestyle=":", alpha=0.7, label=f"Optimal k = {lda_model.optimal_k}")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        ax1.set_title("LDA Model Selection: Coherence & Perplexity vs. k")
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Seeded LDA metrics comparison
        if "Seeded LDA_model" in all_results:
            slda = all_results["Seeded LDA_model"]
            st.markdown("**LDA vs Seeded LDA Metrics Comparison**")
            comp_df = pd.DataFrame([
                {
                    "Model": "LDA (unsupervised)",
                    "k": lda_model.optimal_k,
                    "Perplexity": round(lda_model.perplexity_scores[lda_model.optimal_k], 2),
                    "Coherence (UMass)": round(lda_model.coherence_umass[lda_model.optimal_k], 4),
                    "Coherence (NPMI)": round(lda_model.coherence_npmi[lda_model.optimal_k], 4),
                    "Topic Diversity": round(lda_model.overall_diversity, 4),
                },
                {
                    "Model": "Seeded LDA (guided)",
                    "k": len(CONSTRUCTS),
                    "Perplexity": round(slda.perplexity, 2),
                    "Coherence (UMass)": round(slda.coherence_umass_val, 4),
                    "Coherence (NPMI)": round(slda.coherence_npmi_val, 4),
                    "Topic Diversity": round(slda.topic_diversity_val, 4),
                },
            ])
            st.dataframe(comp_df, use_container_width=True)

            _print_section("LDA vs SEEDED LDA COMPARISON")
            print(comp_df.to_string(index=False))

    # ==================================================================
    # SECTION 2: TOPIC DISTRIBUTION
    # ==================================================================
    st.subheader("2. Topic Distribution")

    dist = topic_distribution(primary_df)
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(dist, use_container_width=True)
    with col2:
        ent = topic_entropy(primary_df)
        st.metric("Topic Entropy (diversity)", ent)
        unclassified_pct = (primary_df["assigned_topic"] == "Unclassified").mean() * 100
        st.metric("Unclassified %", f"{unclassified_pct:.1f}%")

    _print_section(f"TOPIC DISTRIBUTION ({primary_name})")
    print(dist.to_string())
    print(f"\nShannon Entropy: {ent}")
    print(f"Unclassified: {unclassified_pct:.1f}%")

    fig, ax = plt.subplots(figsize=(9, 5))
    dist["count"].plot(kind="barh", ax=ax, color=sns.color_palette("Set2", len(dist)))
    ax.set_xlabel("Count")
    ax.set_title(f"Topic Distribution ({primary_name})")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ==================================================================
    # SECTION 3: CONFIDENCE / ACCURACY PROXIES
    # ==================================================================
    st.subheader("3. Classification Confidence")

    conf = confidence_summary(primary_df)
    st.dataframe(conf, use_container_width=True)

    _print_section(f"CLASSIFICATION CONFIDENCE ({primary_name})")
    print(conf.to_string())

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
        st.dataframe(conf_brand, use_container_width=True)

        _print_section("CONFIDENCE BY BRAND")
        print(conf_brand.to_string())

        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(conf_brand, annot=True, fmt=".3f", cmap="YlGn", ax=ax)
        ax.set_title("Mean Confidence per Brand x Topic")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ==================================================================
    # SECTION 4: KEYWORD ANALYSIS
    # ==================================================================
    st.subheader("4. Keyword Coverage & Strength")

    hit_df = keyword_hit_rate(texts)
    st.dataframe(hit_df, use_container_width=True)

    _print_section("KEYWORD COVERAGE")
    print(hit_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(hit_df["construct"], hit_df["hit_rate_pct"], color="teal")
    ax.set_xlabel("% of documents containing >= 1 keyword")
    ax.set_title("Keyword Hit Rate by Construct")
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Top keywords
    st.subheader("Top Keywords by Construct")
    top_kw = top_keywords_by_construct(texts, top_n=8)
    _print_section("TOP KEYWORDS BY CONSTRUCT")
    for cname, kwlist in top_kw.items():
        with st.expander(f"{cname}"):
            kw_df = pd.DataFrame(kwlist, columns=["Keyword", "Occurrences"])
            st.dataframe(kw_df, use_container_width=True)
        print(f"\n  {cname}:")
        for kw, cnt in kwlist:
            print(f"    {kw:25s} {cnt}")

    # ==================================================================
    # SECTION 5: BRAND COMPARATIVE ANALYSIS
    # ==================================================================
    st.subheader("5. Brand Comparative Analysis")

    bs = brand_summary(primary_df)
    st.dataframe(bs, use_container_width=True)

    _print_section("BRAND COMPARATIVE ANALYSIS")
    print(bs.to_string(index=False))

    # Chi-squared
    chi2 = brand_topic_chi2(primary_df)
    if chi2.get("chi2") is not None:
        st.markdown("**Chi-squared Test: Brand vs. Topic independence**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Chi2", chi2["chi2"])
        col2.metric("p-value", f"{chi2['p_value']:.6f}")
        col3.metric("Significant (p < 0.05)", "Yes" if chi2["significant_at_005"] else "No")

        print(f"\nChi-squared Test:")
        print(f"  Chi2 = {chi2['chi2']}")
        print(f"  p-value = {chi2['p_value']:.6f}")
        print(f"  DOF = {chi2['dof']}")
        print(f"  Significant at 0.05 = {chi2['significant_at_005']}")

    # Brand topic distribution
    pct_df = topic_distribution_by_brand_pct(primary_df)
    if not pct_df.empty:
        st.dataframe(pct_df, use_container_width=True)

        _print_section("TOPIC DISTRIBUTION BY BRAND (%)")
        print(pct_df.to_string())

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
        st.dataframe(be, use_container_width=True)

        _print_section("BRAND TOPIC ENTROPY")
        print(be.to_string(index=False))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(be["brand"], be["topic_entropy"], color=sns.color_palette("Set2"))
        ax.set_ylabel("Shannon Entropy")
        ax.set_title("Topic Diversity per Brand (higher = more diverse)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ==================================================================
    # SECTION 6: CROSS-ALGORITHM AGREEMENT
    # ==================================================================
    if len(method_names) > 1:
        st.subheader("6. Cross-Algorithm Agreement")

        agree_matrix = pairwise_agreement(all_results)
        st.dataframe(agree_matrix, use_container_width=True)

        _print_section("PAIRWISE AGREEMENT MATRIX")
        print(agree_matrix.to_string())

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
            mean_agree = cons["agreement_ratio"].mean()
            full_agree = (cons["agreement_ratio"] == 1.0).mean() * 100
            col1, col2 = st.columns(2)
            col1.metric("Mean agreement ratio", f"{mean_agree:.3f}")
            col2.metric("Full agreement %", f"{full_agree:.1f}%")

            print(f"\nConsensus Statistics:")
            print(f"  Mean agreement ratio: {mean_agree:.3f}")
            print(f"  Full agreement: {full_agree:.1f}%")

            cons_dist = cons["consensus_topic"].value_counts()
            fig, ax = plt.subplots(figsize=(10, 4))
            cons_dist.plot(kind="bar", ax=ax, color=sns.color_palette("Set2"))
            ax.set_title("Consensus Topic Distribution (majority vote)")
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ==================================================================
    # SECTION 7: EXPORT
    # ==================================================================
    st.subheader("7. Export")
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

    _print_section("ALL METRICS PRINTED ABOVE — READY FOR ARTICLE")
    sys.stdout.flush()
