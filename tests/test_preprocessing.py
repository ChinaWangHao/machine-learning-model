import pytest
from spam_detector.preprocessing import preprocess_text, create_vectorizer


def test_preprocess_text_lowercase():
    config = {"lowercase": True}
    text = "Hello World"
    result = preprocess_text(text, config)
    assert result == "hello world"
    

def test_preprocess_text_remove_punctuation():
    config = {"remove_punctuation": True}
    text = "Hello, World!"
    result = preprocess_text(text, config)
    assert result == "Hello World"
    

def test_preprocess_text_combined():
    config = {"lowercase": True, "remove_punctuation": True}
    text = "Hello, World!"
    result = preprocess_text(text, config)
    assert result == "hello world"
    

def test_vectorizer_creation():
    config = {"min_df": 2, "max_features": 1000}
    vectorizer = create_vectorizer(config)
    assert vectorizer.min_df == 2
    assert vectorizer.max_features == 1000
