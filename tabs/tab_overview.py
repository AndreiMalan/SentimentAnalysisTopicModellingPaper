"""
Tab: Data Pipeline & Overview
=============================
Load data, run the cleaning pipeline, display statistics and quality metrics.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pipeline.data_loader import load_comments, dataset_summary
from pipeline.data_cleaner import (
    run_cleaning_pipeline, cleaning_report,
    FTFY_AVAILABLE, LANGDETECT_AVAILABLE, EMOJI_AVAILABLE,
)


def render_tab(uploaded_file=None):
    st.header("Data Pipeline & Overview")
    st.markdown(
        "Load the YouTube comments dataset, inspect raw data, "
        "and run the full cleaning pipeline.  "
        "**Non-English comments are dropped; English comments are cleaned.**"
    )

    # --- Dependency check ---
    with st.expander("Library availability"):
        cols = st.columns(3)
        cols[0].metric("ftfy (encoding)", "Yes" if FTFY_AVAILABLE else "No")
        cols[1].metric("langdetect", "Yes" if LANGDETECT_AVAILABLE else "No")
        cols[2].metric("emoji", "Yes" if EMOJI_AVAILABLE else "No")

    # --- Load data ---
    st.subheader("1. Load Dataset")

    data_path = st.text_input(
        "Path to Comments_and_brands.xlsx",
        value="Comments_and_brands.xlsx",
    )

    if uploaded_file is not None:
        raw_df = pd.read_excel(uploaded_file)
        if "Comment" not in raw_df.columns or "Brand" not in raw_df.columns:
            st.error("Uploaded file must have 'Comment' and 'Brand' columns.")
            return
    else:
        try:
            raw_df = load_comments(data_path)
        except FileNotFoundError as e:
            st.warning(str(e))
            st.info("Upload the file using the sidebar or place it in the project root.")
            return

    st.success(f"Loaded {len(raw_df):,} comments.")
    st.session_state["raw_df"] = raw_df

    # --- Raw data summary ---
    st.subheader("2. Raw Data Summary")
    summary = dataset_summary(raw_df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total comments", f"{summary['total_comments']:,}")
    col2.metric("Unique brands", summary["unique_brands"])
    col3.metric("Avg length (chars)", f"{summary['avg_comment_length']:.0f}")

    # Brand distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    brand_counts = raw_df["Brand"].value_counts()
    colors = sns.color_palette("Set2", len(brand_counts))
    brand_counts.plot(kind="bar", ax=ax, color=colors)
    ax.set_title("Comments per Brand")
    ax.set_ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    with st.expander("Sample raw comments"):
        st.dataframe(raw_df.head(20), use_container_width=True)

    # --- Cleaning Pipeline ---
    st.subheader("3. Run Cleaning Pipeline")
    st.info(
        "Pipeline: fix encoding -> remove emojis -> "
        "detect language -> **keep English only** -> "
        "clean text -> NLP preprocessing"
    )

    if st.button("Run Cleaning Pipeline", type="primary"):
        progress_bar = st.progress(0, text="Starting pipeline...")
        status_text = st.empty()

        def _cb(step, frac):
            progress_bar.progress(min(frac, 1.0), text=f"{step}...")
            status_text.text(f"Step: {step} ({frac * 100:.0f}%)")

        with st.spinner("Running cleaning pipeline..."):
            cleaned_df = run_cleaning_pipeline(
                raw_df,
                progress_callback=_cb,
            )

        progress_bar.progress(1.0, text="Done!")
        status_text.text("Pipeline complete.")

        st.session_state["cleaned_df"] = cleaned_df
        st.success(
            f"Cleaning complete: {len(raw_df):,} -> {len(cleaned_df):,} comments "
            f"({len(raw_df) - len(cleaned_df):,} removed)."
        )

    # --- Show cleaned data ---
    if "cleaned_df" in st.session_state:
        cleaned_df = st.session_state["cleaned_df"]
        report = cleaning_report(raw_df, cleaned_df)

        st.subheader("4. Cleaning Report")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Original", f"{report['original_count']:,}")
        col2.metric("After cleaning", f"{report['cleaned_count']:,}")
        col3.metric("Removed", f"{report['removed_count']:,} ({report['removal_pct']}%)")
        col4.metric("Avg tokens", report["avg_tokens_after"])

        # Brand distribution after cleaning
        st.subheader("Brand Distribution (cleaned)")
        fig, ax = plt.subplots(figsize=(8, 4))
        brand_counts = cleaned_df["Brand"].value_counts()
        brand_counts.plot(kind="bar", ax=ax, color=sns.color_palette("Set2", len(brand_counts)))
        ax.set_title("Comments per Brand (English only, after cleaning)")
        ax.set_ylabel("Count")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        with st.expander("Sample cleaned comments"):
            display_cols = ["Brand", "Comment", "comment_clean", "comment_processed"]
            available = [c for c in display_cols if c in cleaned_df.columns]
            st.dataframe(cleaned_df[available].head(30), use_container_width=True)

        # --- Full cleaned dataset view & download ---
        st.subheader("5. Cleaned Dataset")
        st.markdown(
            "Browse the full cleaned dataset below. "
            "**Original Comment** is shown side-by-side with the **Processed Text**."
        )

        # Build a display DataFrame with the columns the user cares about
        view_cols = ["Brand", "Comment", "comment_processed"]
        available_view = [c for c in view_cols if c in cleaned_df.columns]
        view_df = cleaned_df[available_view].copy()
        view_df.columns = [
            {"Brand": "Brand", "Comment": "Original Comment",
             "comment_processed": "Processed Text"}.get(c, c)
            for c in available_view
        ]

        st.dataframe(view_df, use_container_width=True, height=400)

        # Download buttons
        st.markdown("**Download cleaned dataset:**")
        dl_col1, dl_col2 = st.columns(2)

        # CSV download
        csv_bytes = view_df.to_csv(index=False).encode("utf-8")
        dl_col1.download_button(
            label="Download as CSV",
            data=csv_bytes,
            file_name="cleaned_comments.csv",
            mime="text/csv",
        )

        # Excel download
        from io import BytesIO
        excel_buffer = BytesIO()
        view_df.to_excel(excel_buffer, index=False, engine="openpyxl")
        excel_buffer.seek(0)
        dl_col2.download_button(
            label="Download as Excel",
            data=excel_buffer,
            file_name="cleaned_comments.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
