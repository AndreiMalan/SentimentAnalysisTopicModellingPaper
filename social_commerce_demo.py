"""
SOCIAL COMMERCE TOPIC MODELING - STANDALONE DEMO
=================================================

Run this file to test social commerce topic classification!

Usage:
    streamlit run social_commerce_demo.py

This demo allows you to:
1. Test different classification methods
2. Upload your own CSV file
3. Customize topic definitions
4. Compare methods side-by-side
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import our modules
from guided_topic_modeling import *
from social_commerce_topic_tab import social_commerce_topic_modeling_tab

# ============================================================================
# SAMPLE DATA
# ============================================================================

SAMPLE_SOCIAL_COMMERCE_DATA = [
    # Purchase Intent examples
    "Looking to buy a new laptop, any recommendations under $1000?",
    "I want to order this product, is it available in blue?",
    "Planning to purchase the iPhone 15, should I wait for Black Friday?",
    "Need to buy gifts for Christmas, where can I find good deals?",
    "Going to order this tomorrow if the price stays the same",

    # Product Reviews examples
    "This product is amazing! Best purchase I've made this year!",
    "Terrible quality, broke after 2 days. Don't waste your money.",
    "5 stars! Exactly as described, very satisfied with my purchase.",
    "Disappointed with the quality, not worth the price at all.",
    "Love it! Works perfectly and arrived faster than expected.",

    # Shopping Experience examples
    "Fast delivery! Arrived in just 2 days, well packaged.",
    "Customer service was very helpful when I had issues with my order.",
    "Website is so confusing, couldn't find what I was looking for.",
    "Returned the item easily, no questions asked. Great return policy!",
    "My package was lost and customer support is not responding.",

    # Price Sensitivity examples
    "Too expensive for what it is, I'll wait for a sale.",
    "Great deal! 50% off, couldn't resist buying it.",
    "Is this the best price available? Seems a bit high.",
    "Found a coupon code for 20% off, sharing with everyone!",
    "Not worth full price, but good value on discount.",

    # Social Influence examples
    "Everyone on TikTok is talking about this product!",
    "My friend recommended this and she was absolutely right!",
    "Saw this trending on Instagram, had to get it.",
    "All the reviews say this is the best in its category.",
    "Influencers keep promoting this, wonder if it's actually good.",

    # Mixed examples
    "Just bought this based on recommendations. Hope it's as good as they say! A bit expensive though.",
    "Delivery was super fast but the product quality is disappointing.",
    "Great customer service helped me get a discount, very happy!",
    "Ordered this because everyone has one, but shipping took forever.",
    "Looking for cheaper alternatives to this popular product.",
] * 20  # Repeat to have more data

# Create DataFrame
SAMPLE_DF = pd.DataFrame({
    'text': SAMPLE_SOCIAL_COMMERCE_DATA,
    'sentiment': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0] * 20
})

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="Social Commerce Topic Modeling",
        page_icon="🛒",
        layout="wide"
    )

    st.title("🛒 Social Commerce Topic Modeling Demo")
    st.markdown("""
    This tool classifies online posts and comments into **social commerce behavior categories**.

    **Available Methods:**
    - 🔑 **Keyword-Based**: Fast, interpretable (instant results)
    - 📊 **Guided LDA**: Probabilistic topic discovery (1-2 min)
    - 🤖 **Zero-Shot Transformer**: Highest accuracy (5-10 min)
    - 🧠 **BERTopic**: Best topic quality (10-15 min)
    """)

    # Sidebar: Data source selection
    st.sidebar.header("📁 Data Source")
    data_source = st.sidebar.radio(
        "Choose data:",
        ["Use Sample Data", "Upload CSV File"]
    )

    if data_source == "Use Sample Data":
        st.sidebar.success(f"✅ Using sample data ({len(SAMPLE_DF)} documents)")
        data = SAMPLE_DF

        # Show sample data preview
        with st.sidebar.expander("👀 Preview Sample Data"):
            st.dataframe(data.head(10))

    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload your CSV file",
            type=['csv'],
            help="CSV must have 'text' column. Optional 'sentiment' column."
        )

        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)

                # Validate columns
                if 'text' not in data.columns:
                    st.error("❌ CSV must have a 'text' column!")
                    return

                # Add sentiment column if missing
                if 'sentiment' not in data.columns:
                    data['sentiment'] = 0  # Placeholder

                st.sidebar.success(f"✅ Loaded {len(data)} documents")

                with st.sidebar.expander("👀 Preview Your Data"):
                    st.dataframe(data.head(10))

            except Exception as e:
                st.error(f"Error loading file: {e}")
                return
        else:
            st.info("👈 Upload a CSV file to get started!")
            return

    # Main content
    tabs = st.tabs([
        "📊 Topic Classification",
        "📝 Quick Test",
        "ℹ️ About Topics"
    ])

    # Tab 1: Main topic classification
    with tabs[0]:
        social_commerce_topic_modeling_tab(data)

    # Tab 2: Quick test
    with tabs[1]:
        st.header("📝 Quick Single Text Test")
        st.write("Test classification on individual texts")

        # Example texts
        example_texts = {
            "Purchase Intent": "I'm looking to buy a new phone. Any good deals available?",
            "Product Review": "This product is amazing! Best quality, highly recommend to everyone!",
            "Shopping Experience": "Fast delivery, great customer service, very satisfied!",
            "Price Sensitivity": "Too expensive! I'll wait for a discount or sale.",
            "Social Influence": "Everyone on social media is buying this, must be good!"
        }

        selected_example = st.selectbox(
            "Choose an example or write your own:",
            ["Custom"] + list(example_texts.keys())
        )

        if selected_example == "Custom":
            test_text = st.text_area(
                "Enter your text:",
                height=100,
                placeholder="Type or paste text here..."
            )
        else:
            test_text = st.text_area(
                "Enter your text:",
                value=example_texts[selected_example],
                height=100
            )

        col1, col2 = st.columns(2)
        with col1:
            test_method = st.radio(
                "Classification Method:",
                ["Keyword-Based (Fast)", "Zero-Shot (Accurate)"]
            )

        if st.button("🔍 Classify Text", type="primary"):
            if not test_text.strip():
                st.warning("Please enter some text!")
            else:
                if test_method == "Keyword-Based (Fast)":
                    classifier = KeywordTopicClassifier(SOCIAL_COMMERCE_TOPICS)
                    topic, confidence, scores = classifier.classify_document(test_text)

                    # Display result
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"**Predicted Topic:** {topic}")
                    with col2:
                        st.metric("Confidence", f"{confidence:.1%}")

                    # All scores
                    st.markdown("### 📊 All Topic Scores")
                    for t, s in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                        st.progress(s, text=f"{t}: {s:.1%}")

                else:  # Zero-Shot
                    if not TRANSFORMERS_AVAILABLE:
                        st.error("Transformers not installed! Use: pip install transformers torch")
                    else:
                        with st.spinner("Classifying with transformer model..."):
                            classifier = ZeroShotTopicClassifier(SOCIAL_COMMERCE_TOPICS)
                            topic, confidence, scores = classifier.classify_document(test_text)

                            # Display result
                            col1, col2 = st.columns(2)
                            with col1:
                                st.success(f"**Predicted Topic:** {topic}")
                            with col2:
                                st.metric("Confidence", f"{confidence:.1%}")

                            # All scores
                            st.markdown("### 📊 All Topic Scores")
                            for t, s in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                                st.progress(s, text=f"{t}: {s:.1%}")

    # Tab 3: About topics
    with tabs[2]:
        st.header("ℹ️ Social Commerce Topic Definitions")

        for topic_name, topic_info in SOCIAL_COMMERCE_TOPICS.items():
            with st.expander(f"📌 {topic_name}"):
                st.markdown(f"**Description:** {topic_info['description']}")

                st.markdown("**Seed Words (Keywords):**")
                st.write(", ".join(topic_info['seed_words'][:20]))

                st.markdown("**Example Phrases:**")
                # Generate example based on seed words
                if topic_name == "Purchase Intent":
                    examples = [
                        "- Looking to buy...",
                        "- Want to order...",
                        "- Planning to purchase...",
                        "- Need to get..."
                    ]
                elif topic_name == "Product Reviews":
                    examples = [
                        "- This is amazing!",
                        "- Terrible quality",
                        "- 5 stars, highly recommend",
                        "- Not worth the money"
                    ]
                elif topic_name == "Shopping Experience":
                    examples = [
                        "- Fast delivery",
                        "- Great customer service",
                        "- Easy to navigate website",
                        "- Package was delayed"
                    ]
                elif topic_name == "Price Sensitivity":
                    examples = [
                        "- Too expensive",
                        "- Great deal!",
                        "- 50% off",
                        "- Worth the price"
                    ]
                else:  # Social Influence
                    examples = [
                        "- Everyone recommends this",
                        "- Trending on social media",
                        "- My friend suggested this",
                        "- Influencers love it"
                    ]

                for ex in examples:
                    st.write(ex)

    # Footer
    st.markdown("---")
    st.markdown("""
    ### 📚 Next Steps

    1. **Test with your data**: Upload your CSV file with online comments/posts
    2. **Choose method**: Start with Keyword-Based for speed, use Zero-Shot for accuracy
    3. **Analyze results**: Look at topic distribution and confidence scores
    4. **Refine topics**: Edit seed words if needed for better classification
    5. **Combine with sentiment**: Use classified topics + sentiment analysis together!

    ### 💡 Tips

    - **Keyword-Based** is best for quick exploration and validation
    - **Zero-Shot Transformer** gives best accuracy but needs GPU for speed
    - **Guided LDA** is good for discovering new relevant keywords
    - **BERTopic** provides highest quality topics but is slowest

    ### ❓ Questions?

    - How to customize topics? → Edit SOCIAL_COMMERCE_TOPICS in guided_topic_modeling.py
    - Need more topics? → Add new categories with seed words
    - Want supervised learning? → Label some data and train a classifier
    """)


if __name__ == "__main__":
    main()
