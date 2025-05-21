import pandas as pd
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split

def load_data(file_path: str) -> pd.DataFrame:
    """Load email data from a CSV file.
    
    Args:
        file_path: Path to the CSV file containing email data
        
    Returns:
        DataFrame containing email data
    """
    return pd.read_csv(file_path)

def split_data(data: pd.DataFrame, text_column: str, label_column: str, 
              test_size: float = 0.2, random_state: Optional[int] = None) -> Tuple[List[str], List[str], List[int], List[int]]:
    """Split data into training and testing sets.
    
    Args:
        data: DataFrame containing email data
        text_column: Name of the column containing email text
        label_column: Name of the column containing spam/ham labels
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X = data[text_column].tolist()
    y = data[label_column].tolist()
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
