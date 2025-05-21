from typing import Dict, Any, List

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator

def create_model(config: Dict[str, Any]) -> BaseEstimator:
    """Create a model based on configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        Scikit-learn model
    """
    model_type = config['model']['type']
    
    if model_type == 'naive_bayes':
        nb_config = config.get('naive_bayes', {})
        return MultinomialNB(
            alpha=nb_config.get('alpha', 1.0),
            fit_prior=nb_config.get('fit_prior', True)
        )
    
    elif model_type == 'svm':
        svm_config = config.get('svm', {})
        return SVC(
            C=svm_config.get('C', 1.0),
            kernel=svm_config.get('kernel', 'linear'),
            gamma=svm_config.get('gamma', 'scale'),
            probability=True
        )
    
    elif model_type == 'random_forest':
        rf_config = config.get('random_forest', {})
        return RandomForestClassifier(
            n_estimators=rf_config.get('n_estimators', 100),
            max_depth=rf_config.get('max_depth'),
            min_samples_split=rf_config.get('min_samples_split', 2),
            min_samples_leaf=rf_config.get('min_samples_leaf', 1),
            random_state=config['data'].get('random_state')
        )
    
    elif model_type == 'logistic_regression':
        lr_config = config.get('logistic_regression', {})
        return LogisticRegression(
            C=lr_config.get('C', 1.0),
            penalty=lr_config.get('penalty', 'l2'),
            solver=lr_config.get('solver', 'lbfgs'),
            max_iter=lr_config.get('max_iter', 100),
            random_state=config['data'].get('random_state')
        )
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
