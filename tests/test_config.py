import pytest
import tempfile
import os
from spam_detector.config import load_config


def test_load_config():
    # Create a temporary TOML config file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.toml') as f:
        f.write("""
        [data]
        train_test_split = 0.2
        random_state = 42
        
        [preprocessing]
        min_df = 5
        max_features = 10000
        stop_words = "english"
        
        [model]
        type = "naive_bayes"
        """)
        temp_file = f.name
    
    try:
        # Load the config
        config = load_config(temp_file)
        
        # Check if config loaded correctly
        assert config['data']['train_test_split'] == 0.2
        assert config['data']['random_state'] == 42
        assert config['preprocessing']['min_df'] == 5
        assert config['preprocessing']['max_features'] == 10000
        assert config['preprocessing']['stop_words'] == "english"
        assert config['model']['type'] == "naive_bayes"
    finally:
        # Clean up temporary file
        os.unlink(temp_file)
