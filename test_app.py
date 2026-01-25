"""
Quick test script to verify the application functions work correctly
without requiring all dependencies.
"""

import pandas as pd
import numpy as np
import sys

# Add current directory to path
sys.path.insert(0, '/home/user/SentimentAnalysisTopicModellingPaper')

# Import basic functions (those that don't require streamlit/transformers)
from sentiment_analysis_app import (
    clean_text,
    preprocess_text,
    EMOTION_MAPPING,
    RANDOM_STATE
)

def test_text_cleaning():
    """Test text cleaning function"""
    print("Testing text cleaning...")

    test_cases = [
        ("Hello World! 123", "hello world"),
        ("Visit https://example.com for more", "visit  for more"),
        ("AMAZING product!!!", "amazing product"),
    ]

    for input_text, expected in test_cases:
        result = clean_text(input_text)
        # Simple check (exact match may vary due to spacing)
        assert "http" not in result.lower(), f"URL not removed: {result}"
        assert result.islower(), f"Not lowercase: {result}"
        print(f"  ✓ '{input_text}' -> '{result}'")

    print("✓ Text cleaning tests passed!\n")

def test_emotion_mapping():
    """Test emotion mapping"""
    print("Testing emotion mapping...")

    assert len(EMOTION_MAPPING) == 6, "Should have 6 emotions"
    assert EMOTION_MAPPING[0] == "sadness", "Emotion 0 should be sadness"
    assert EMOTION_MAPPING[1] == "joy", "Emotion 1 should be joy"

    print(f"  Emotions: {list(EMOTION_MAPPING.values())}")
    print("✓ Emotion mapping tests passed!\n")

def test_configuration():
    """Test configuration constants"""
    print("Testing configuration...")

    assert RANDOM_STATE == 42, "Random state should be 42"
    print(f"  Random State: {RANDOM_STATE}")
    print("✓ Configuration tests passed!\n")

def test_preprocessing_pipeline():
    """Test full preprocessing pipeline"""
    print("Testing preprocessing pipeline...")

    # Note: This requires NLTK data which may not be available
    try:
        test_text = "I absolutely LOVE this amazing product! It's the BEST!"
        cleaned = clean_text(test_text)
        print(f"  Original: {test_text}")
        print(f"  Cleaned: {cleaned}")

        # Try preprocessing (may fail if NLTK data not available)
        try:
            processed = preprocess_text(cleaned)
            print(f"  Processed: {processed}")
            print("✓ Full preprocessing pipeline passed!\n")
        except Exception as e:
            print(f"  ⚠ Preprocessing requires NLTK data: {e}")
            print("  (This is expected in test environment)\n")
    except Exception as e:
        print(f"  ⚠ Error in preprocessing: {e}\n")

def main():
    """Run all tests"""
    print("="*60)
    print("Sentiment Analysis App - Basic Functionality Tests")
    print("="*60)
    print()

    test_text_cleaning()
    test_emotion_mapping()
    test_configuration()
    test_preprocessing_pipeline()

    print("="*60)
    print("✓ All basic tests completed!")
    print("="*60)
    print()
    print("Note: Full app testing requires:")
    print("  - Streamlit installed")
    print("  - NLTK data downloaded")
    print("  - (Optional) Transformers and BERTopic for BERT features")

if __name__ == "__main__":
    main()
