"""
Social Commerce Topic Modeling Tab for Streamlit
=================================================

This module adds a comprehensive topic modeling interface specifically
for social commerce behavior classification.

Integrate this into your main Streamlit app.
"""

import streamlit as st
import pandas as pd
import numpy as np
from guided_topic_modeling import (
    SOCIAL_COMMERCE_TOPICS,
    KeywordTopicClassifier,
    GuidedLDATopicModel,
    ZeroShotTopicClassifier,
    GuidedBERTopicModel,
    plot_topic_distribution,
    plot_confidence_distribution,
    plot_topic_heatmap,
    compare_methods,
    TRANSFORMERS_AVAILABLE,
    BERTOPIC_AVAILABLE
)

def social_commerce_topic_modeling_tab(data):
    """
    Main function for social commerce topic modeling tab

    Args:
        data: DataFrame with 'text' and 'sentiment' columns
    """
    st.header("📊 Social Commerce Topic Modeling")
    st.markdown("""
    Classify your online posts/comments into **5 social commerce behavior categories**:
    - 🛒 **Purchase Intent** - Users looking to buy
    - ⭐ **Product Reviews** - Quality feedback and ratings
    - 🚚 **Shopping Experience** - Delivery, customer service
    - 💰 **Price Sensitivity** - Deals, discounts, value
    - 👥 **Social Influence** - Recommendations, trending
    """)

    # ========================================================================
    # SECTION 1: Topic Definitions (Customizable)
    # ========================================================================

    with st.expander("📋 Topic Definitions & Seed Words", expanded=False):
        st.markdown("### Customize Your Social Commerce Topics")
        st.info("These seed words guide the topic modeling. You can edit them!")

        for topic_name, topic_info in SOCIAL_COMMERCE_TOPICS.items():
            st.markdown(f"#### {topic_name}")
            st.write(f"**Description:** {topic_info['description']}")

            # Show seed words
            seed_words_str = ", ".join(topic_info['seed_words'][:15])
            st.text(f"Seed words: {seed_words_str}...")

            # Allow editing (stored in session state)
            if f"custom_seeds_{topic_name}" not in st.session_state:
                st.session_state[f"custom_seeds_{topic_name}"] = topic_info['seed_words']

    # ========================================================================
    # SECTION 2: Method Selection
    # ========================================================================

    st.markdown("---")
    st.subheader("🎯 Choose Classification Method")

    method = st.radio(
        "Select approach:",
        [
            "🔑 Keyword-Based (Fastest, Interpretable)",
            "📊 Guided LDA (Balanced, Probabilistic)",
            "🤖 Zero-Shot Transformer (Most Accurate, Slower)" if TRANSFORMERS_AVAILABLE else None,
            "🧠 BERTopic (Best Quality, Slowest)" if BERTOPIC_AVAILABLE else None,
            "🔬 Compare All Methods"
        ],
        help="Each method has different speed/accuracy tradeoffs"
    )

    # Remove None options
    if method is None:
        st.warning("Some methods require additional libraries. Install transformers/bertopic for more options.")
        return

    # ========================================================================
    # SECTION 3: Run Classification
    # ========================================================================

    st.markdown("---")

    # Sample size selector
    col1, col2 = st.columns(2)
    with col1:
        sample_size = st.slider(
            "Number of documents to classify:",
            min_value=100,
            max_value=min(10000, len(data)),
            value=min(1000, len(data)),
            step=100
        )
    with col2:
        random_sample = st.checkbox("Random sample", value=True)

    # Get sample
    if random_sample:
        sample_data = data.sample(n=sample_size, random_state=42)
    else:
        sample_data = data.head(sample_size)

    texts = sample_data['text'].tolist()

    # Run button
    if st.button("🚀 Classify Topics", type="primary"):

        # =================================================================
        # METHOD 1: KEYWORD-BASED
        # =================================================================
        if "Keyword-Based" in method:
            st.subheader("🔑 Keyword-Based Classification")

            with st.spinner("Classifying with keyword matching..."):
                classifier = KeywordTopicClassifier(SOCIAL_COMMERCE_TOPICS)
                results = classifier.classify_corpus(texts)

            # Display results
            st.success(f"✅ Classified {len(results)} documents!")

            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_conf = results['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_conf:.2%}")
            with col2:
                known_topics = len(results[results['predicted_topic'] != 'Unknown/Other'])
                st.metric("Known Topics", f"{known_topics}/{len(results)}")
            with col3:
                unique_topics = results['predicted_topic'].nunique()
                st.metric("Unique Topics", unique_topics)

            # Topic distribution
            st.markdown("### 📊 Topic Distribution")
            fig = plot_topic_distribution(results, "Social Commerce Topics Distribution")
            st.pyplot(fig)

            # Confidence distribution
            st.markdown("### 📈 Confidence Distribution")
            fig = plot_confidence_distribution(results)
            st.pyplot(fig)

            # Sample results table
            st.markdown("### 📋 Sample Classifications")
            display_cols = ['text', 'predicted_topic', 'confidence']
            st.dataframe(
                results[display_cols].head(20),
                use_container_width=True
            )

            # Topic-wise examples
            st.markdown("### 📝 Examples by Topic")
            for topic in results['predicted_topic'].unique():
                if topic == 'Unknown/Other':
                    continue

                with st.expander(f"Examples: {topic}"):
                    topic_examples = results[results['predicted_topic'] == topic].head(5)
                    for _, row in topic_examples.iterrows():
                        st.write(f"- {row['text']} (confidence: {row['confidence']:.2%})")

            # Heatmap
            st.markdown("### 🔥 Document-Topic Score Heatmap")
            fig = plot_topic_heatmap(results, SOCIAL_COMMERCE_TOPICS)
            if fig:
                st.pyplot(fig)

            # Download results
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Results (CSV)",
                csv,
                "social_commerce_topics.csv",
                "text/csv"
            )

        # =================================================================
        # METHOD 2: GUIDED LDA
        # =================================================================
        elif "Guided LDA" in method:
            st.subheader("📊 Guided LDA Topic Modeling")

            with st.spinner("Training Guided LDA model... (may take 1-2 minutes)"):
                classifier = GuidedLDATopicModel(SOCIAL_COMMERCE_TOPICS, n_topics=5)
                classifier.fit(texts)
                results = classifier.predict(texts)

            st.success(f"✅ Classified {len(results)} documents!")

            # Show discovered topics
            st.markdown("### 🔍 Discovered Topics (LDA)")
            topic_words = classifier.get_topic_words(n_words=15)

            for topic_name, words in topic_words.items():
                with st.expander(topic_name):
                    st.write(f"**Top words:** {', '.join(words)}")

            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_conf = results['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_conf:.2%}")
            with col2:
                known_topics = len(results[results['predicted_topic'] != 'Unknown/Other'])
                st.metric("Known Topics", f"{known_topics}/{len(results)}")
            with col3:
                unique_topics = results['predicted_topic'].nunique()
                st.metric("Unique Topics", unique_topics)

            # Topic distribution
            st.markdown("### 📊 Topic Distribution")
            fig = plot_topic_distribution(results, "Guided LDA Topics")
            st.pyplot(fig)

            # Sample results
            st.markdown("### 📋 Sample Classifications")
            st.dataframe(
                results[['text', 'predicted_topic', 'confidence']].head(20),
                use_container_width=True
            )

            # Download
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Results (CSV)",
                csv,
                "guided_lda_topics.csv",
                "text/csv"
            )

        # =================================================================
        # METHOD 3: ZERO-SHOT TRANSFORMER
        # =================================================================
        elif "Zero-Shot" in method:
            if not TRANSFORMERS_AVAILABLE:
                st.error("Transformers library required! Install with: pip install transformers torch")
                return

            st.subheader("🤖 Zero-Shot Transformer Classification")
            st.info("⏱️ This may take 5-10 minutes for 1000 documents (faster with GPU)")

            # Option to use descriptions
            use_descriptions = st.checkbox(
                "Use topic descriptions for classification",
                value=True,
                help="Uses full topic descriptions instead of just names for better accuracy"
            )

            with st.spinner("Loading transformer model and classifying... Please wait..."):
                try:
                    classifier = ZeroShotTopicClassifier(SOCIAL_COMMERCE_TOPICS)
                    results = classifier.classify_corpus(texts, use_descriptions=use_descriptions)

                    st.success(f"✅ Classified {len(results)} documents!")

                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        avg_conf = results['confidence'].mean()
                        st.metric("Avg Confidence", f"{avg_conf:.2%}")
                    with col2:
                        high_conf = len(results[results['confidence'] > 0.7])
                        st.metric("High Confidence (>70%)", f"{high_conf}/{len(results)}")
                    with col3:
                        unique_topics = results['predicted_topic'].nunique()
                        st.metric("Unique Topics", unique_topics)

                    # Topic distribution
                    st.markdown("### 📊 Topic Distribution")
                    fig = plot_topic_distribution(results, "Zero-Shot Classification")
                    st.pyplot(fig)

                    # Confidence distribution
                    st.markdown("### 📈 Confidence by Topic")
                    fig = plot_confidence_distribution(results)
                    st.pyplot(fig)

                    # Sample results
                    st.markdown("### 📋 High Confidence Predictions")
                    high_conf_results = results[results['confidence'] > 0.7].head(20)
                    st.dataframe(
                        high_conf_results[['text', 'predicted_topic', 'confidence']],
                        use_container_width=True
                    )

                    # Heatmap
                    st.markdown("### 🔥 Topic Probability Heatmap")
                    fig = plot_topic_heatmap(results, SOCIAL_COMMERCE_TOPICS)
                    if fig:
                        st.pyplot(fig)

                    # Download
                    csv = results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Results (CSV)",
                        csv,
                        "zeroshot_topics.csv",
                        "text/csv"
                    )

                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())

        # =================================================================
        # METHOD 4: BERTOPIC
        # =================================================================
        elif "BERTopic" in method:
            if not BERTOPIC_AVAILABLE:
                st.error("BERTopic library required! Install with: pip install bertopic sentence-transformers")
                return

            st.subheader("🧠 BERTopic with Guided Mapping")
            st.info("⏱️ This may take 10-15 minutes for 1000 documents")

            with st.spinner("Running BERTopic... Please wait..."):
                try:
                    classifier = GuidedBERTopicModel(SOCIAL_COMMERCE_TOPICS)
                    classifier.fit(texts)
                    results = classifier.predict(texts)

                    st.success(f"✅ Classified {len(results)} documents!")

                    # Show topic info
                    st.markdown("### 🔍 Discovered Topics")
                    topic_info = classifier.get_topic_info()
                    st.dataframe(topic_info[['Topic', 'Count', 'Name', 'Mapped_Category']].head(10))

                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        avg_conf = results['confidence'].mean()
                        st.metric("Avg Confidence", f"{avg_conf:.2%}")
                    with col2:
                        known_topics = len(results[results['predicted_topic'] != 'Unknown/Other'])
                        st.metric("Known Topics", f"{known_topics}/{len(results)}")
                    with col3:
                        unique_topics = results['predicted_topic'].nunique()
                        st.metric("Unique Topics", unique_topics)

                    # Topic distribution
                    st.markdown("### 📊 Topic Distribution")
                    fig = plot_topic_distribution(results, "BERTopic Classification")
                    st.pyplot(fig)

                    # Sample results
                    st.markdown("### 📋 Sample Classifications")
                    st.dataframe(
                        results[['text', 'predicted_topic', 'confidence']].head(20),
                        use_container_width=True
                    )

                    # Download
                    csv = results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Results (CSV)",
                        csv,
                        "bertopic_classification.csv",
                        "text/csv"
                    )

                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())

        # =================================================================
        # METHOD 5: COMPARE ALL
        # =================================================================
        elif "Compare All" in method:
            st.subheader("🔬 Method Comparison")
            st.warning("⏱️ This will run all available methods. May take 15-20 minutes.")

            # Limit sample size for comparison
            compare_size = min(500, len(texts))
            st.info(f"Comparing methods on {compare_size} documents")

            compare_texts = texts[:compare_size]

            if st.button("🚀 Run Comparison", type="primary"):
                with st.spinner("Running all methods... Please wait..."):
                    comparison_results = compare_methods(compare_texts, SOCIAL_COMMERCE_TOPICS)

                st.success("✅ Comparison complete!")

                # Summary statistics by method
                st.markdown("### 📊 Method Comparison Summary")

                summary_stats = comparison_results.groupby('method').agg({
                    'confidence': ['mean', 'std'],
                    'predicted_topic': lambda x: x.value_counts().index[0]  # Most common topic
                }).round(3)

                st.dataframe(summary_stats)

                # Agreement matrix
                st.markdown("### 🤝 Method Agreement")

                # Pivot for agreement analysis
                pivot_data = comparison_results.pivot_table(
                    index='text',
                    columns='method',
                    values='predicted_topic',
                    aggfunc='first'
                )

                # Calculate pairwise agreement
                methods = pivot_data.columns
                agreement_matrix = pd.DataFrame(
                    index=methods,
                    columns=methods,
                    dtype=float
                )

                for m1 in methods:
                    for m2 in methods:
                        if m1 == m2:
                            agreement_matrix.loc[m1, m2] = 1.0
                        else:
                            agreement = (pivot_data[m1] == pivot_data[m2]).mean()
                            agreement_matrix.loc[m1, m2] = agreement

                st.dataframe(agreement_matrix.astype(float).round(2))

                # Detailed results
                st.markdown("### 📋 Detailed Results (Sample)")
                sample_comparison = comparison_results.head(30)
                st.dataframe(sample_comparison)

                # Download
                csv = comparison_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Comparison (CSV)",
                    csv,
                    "method_comparison.csv",
                    "text/csv"
                )

    # ========================================================================
    # SECTION 4: Single Text Testing
    # ========================================================================

    st.markdown("---")
    st.subheader("🎯 Test Single Text")
    st.write("Quickly test classification on a single text")

    test_text = st.text_area(
        "Enter text to classify:",
        value="Looking to buy a new laptop. Anyone have recommendations for good deals?",
        height=100
    )

    test_method = st.selectbox(
        "Method:",
        ["Keyword-Based", "Zero-Shot" if TRANSFORMERS_AVAILABLE else None]
    )

    if st.button("🔍 Classify Text"):
        if test_method == "Keyword-Based":
            classifier = KeywordTopicClassifier(SOCIAL_COMMERCE_TOPICS)
            topic, confidence, scores = classifier.classify_document(test_text)

            st.success(f"**Predicted Topic:** {topic}")
            st.metric("Confidence", f"{confidence:.2%}")

            st.markdown("### All Topic Scores:")
            for t, s in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                st.progress(s, text=f"{t}: {s:.2%}")

        elif test_method == "Zero-Shot" and TRANSFORMERS_AVAILABLE:
            with st.spinner("Classifying with transformer..."):
                classifier = ZeroShotTopicClassifier(SOCIAL_COMMERCE_TOPICS)
                topic, confidence, scores = classifier.classify_document(test_text)

                st.success(f"**Predicted Topic:** {topic}")
                st.metric("Confidence", f"{confidence:.2%}")

                st.markdown("### All Topic Scores:")
                for t, s in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                    st.progress(s, text=f"{t}: {s:.2%}")

    # ========================================================================
    # SECTION 5: Recommendations
    # ========================================================================

    st.markdown("---")
    with st.expander("💡 Method Recommendations", expanded=False):
        st.markdown("""
        ### Which Method Should You Use?

        **🔑 Keyword-Based**
        - ✅ Use when: Need fast results, have good seed words, want interpretability
        - ⚠️ Limitations: Simple matching, may miss nuanced expressions
        - ⏱️ Speed: ~1 second for 1000 documents

        **📊 Guided LDA**
        - ✅ Use when: Want probabilistic framework, need topic discovery + guidance
        - ⚠️ Limitations: Requires tuning, slower than keywords
        - ⏱️ Speed: ~1-2 minutes for 1000 documents

        **🤖 Zero-Shot Transformer**
        - ✅ Use when: Need highest accuracy, have clear topic definitions, have GPU
        - ⚠️ Limitations: Slower, requires more memory
        - ⏱️ Speed: ~5-10 minutes for 1000 documents (CPU), ~2 minutes (GPU)

        **🧠 BERTopic**
        - ✅ Use when: Need best topic coherence, want semantic understanding
        - ⚠️ Limitations: Slowest, most complex
        - ⏱️ Speed: ~10-15 minutes for 1000 documents

        ### Recommended Workflow:

        1. **Start with Keyword-Based** - Quick prototyping, validate seed words
        2. **Try Zero-Shot** - Best accuracy for classification
        3. **Use Guided LDA** - Discover new relevant words, probabilistic framework
        4. **BERTopic for Production** - Best quality for final deployment

        ### Next Step: Combine with Sentiment!

        Once you have topic classifications, you can:
        - Analyze sentiment per topic
        - Discover which topics have negative/positive sentiment
        - Track sentiment trends for each topic over time
        """)


if __name__ == "__main__":
    # Standalone testing
    import pandas as pd

    # Create sample data
    sample_data = pd.DataFrame({
        'text': [
            "I want to buy a new phone. Any recommendations?",
            "Just received my order! Quality is amazing!",
            "Terrible customer service. Package delayed by 2 weeks.",
            "Is this worth the price? Seems expensive.",
            "Everyone on social media is buying this!",
        ] * 20,  # Repeat for more samples
        'sentiment': [1, 1, 0, 0, 1] * 20
    })

    social_commerce_topic_modeling_tab(sample_data)
