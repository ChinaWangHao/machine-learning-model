from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator

def evaluate_model(model: BaseEstimator, X_train: np.ndarray, X_test: np.ndarray, 
                 y_train: List[int], y_test: List[int], config: Dict[str, Any]) -> Dict[str, float]:
    """Evaluate a model using various metrics.
    
    Args:
        model: Trained model
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
        config: Evaluation configuration
        
    Returns:
        Dictionary of metric names and values
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    results = {}
    metrics = config['evaluation'].get('metrics', ['accuracy'])
    
    if 'accuracy' in metrics:
        results['accuracy'] = accuracy_score(y_test, y_pred)
    
    if 'precision' in metrics:
        results['precision'] = precision_score(y_test, y_pred, zero_division=0)
    
    if 'recall' in metrics:
        results['recall'] = recall_score(y_test, y_pred)
    
    if 'f1' in metrics:
        results['f1'] = f1_score(y_test, y_pred)
    
    if 'roc_auc' in metrics and y_pred_proba is not None:
        results['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    
    # Cross-validation
    cv_folds = config['model'].get('cross_validation_folds', 5)
    cv_score = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
    results['cv_accuracy_mean'] = cv_score.mean()
    results['cv_accuracy_std'] = cv_score.std()
    
    return results

def plot_confusion_matrix(y_true: List[int], y_pred: List[int], save_path: str = None) -> None:
    """Plot a confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Ham', 'Spam']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
