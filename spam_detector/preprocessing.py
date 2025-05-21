import re
import string
from typing import List, Dict, Any, Union
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(text: str, config: Dict[str, Any]) -> str:
    """Preprocess a single text document.
    
    Args:
        text: Input text
        config: Preprocessing configuration
        
    Returns:
        Preprocessed text
    """
    # Convert to lowercase if specified
    if config.get('lowercase', True):
        text = text.lower()
    
    # Remove punctuation if specified
    if config.get('remove_punctuation', True):
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = text.split()
    
    # Remove stop words if specified
    if config.get('stop_words'):
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words(config['stop_words']))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Apply stemming if specified
    if config.get('stem_words', False):
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    
    # Join tokens back into a string
    return ' '.join(tokens)

def create_vectorizer(config: Dict[str, Any]) -> TfidfVectorizer:
    """Create a TF-IDF vectorizer based on configuration.
    
    Args:
        config: Preprocessing configuration
        
    Returns:
        Configured TF-IDF vectorizer
    """
    return TfidfVectorizer(
        min_df=config.get('min_df', 5),
        max_features=config.get('max_features', 10000),
        stop_words=config.get('stop_words', 'english') if config.get('stop_words') else None
    )
